class ModelMerge(nn.Module):
    """
    Handles all merge operations for zipping arbitrary numbers of models. 
    Expects a list of architecture graphs (one per model) (See graphs/base_graphs.py)).
    """
    def __init__(self, *graphs, device=0):
        super().__init__()
        
        self.stop_at = None
        self.start_at = None
        self.stop_at_ptr = [None]
        self.start_at_ptr = {}

        self.hooks = []

        self.init(graphs, device)

    def init(self, graphs, device):
        """
        Initialize merge attributes with new set of graphs.
        """
        # move all graph models to eval
        for g in graphs:
            g.model.to(device).eval()

        self.graphs = graphs
        self.device = device

        self.merged_model = None
        # Initialize heads for partial zipping
        self.head_models = nn.ModuleList([g.model for g in self.graphs])
        # Add hooks on intermediate layers for computing intra-model alignment metrics
        for graph in self.graphs:
            graph.add_hooks(device=device)


    def compute_metrics(self, dataloader, metric_classes):
        """
        Compute pairwise alignment metrics between all graph models (self inclusive).
        - dataloader: pytorch dataloader. Dataset (or list of datasets) over which to compute metrics
        - metric_classes: dictionary whose keys are metric types, and values are metric functions.
            This function will compute metrics for all kinds in metric_classes, using the dataloader.
        
        This function performs a forward pass over the dataset, aggregating all intermediate representations
        among all hooks in a graph model. These are then combined to calculate metrics.    
        
        Returns: dictionary of graph nodes to metrics computed at those nodes in the model graph.
        """
        self.metrics = None
        if not isinstance(dataloader, list):
            dataloader_list = [dataloader]
        else:
            dataloader_list = dataloader
        
        numel = 0
        for dataloader in dataloader_list:
            for x, _ in tqdm(dataloader, desc="Forward Pass to Compute Merge Metrics: "):
                x = x.to(self.device)
                
                numel += x.shape[0]
                intermediates = [g.compute_intermediates(x) for g in self.graphs]
                nodes = list(intermediates[0].keys())
                if self.metrics is None:
                    self.metrics = {n: {k: v() for k, v in metric_classes.items()} for n in nodes}
                
                for node, node_metrics in self.metrics.items():
                    for metric in node_metrics.values():
                        intermeds_float = [i[node].float() for i in intermediates]
                        metric.update(x.shape[0], *intermeds_float)
        
        for node, node_metrics in self.metrics.items():
            for metric_name, metric in node_metrics.items():
                self.metrics[node][metric_name] = metric.finalize(numel)

        return self.metrics
    
    def compute_transformations(self, transform_fn, reduce_ratio=.5, **kwargs):
        """
        Transforms graph models according to a transform function (transform_fn) using the alignment 
        metrics provided by self.metrics. Will transform the feature spaces at each PREFIX and POSTFIX 
        node between all models. The objective of this operation is to map all dispirate feature spaces 
        in each model graph to a common one such that all distinct spaces collectively map to a single 
        space of dimension (1 - reduce_ratio) * sum(graph1_feat_dim + graph2_feat_dim + ... + graphN_feat_dim)
        - transform_fn: transformation function (e.g., permutation - match_tensors_permute)
        - reduce_ratio: desired reduction proportion from total of all graph model feature dimensions
        - kwargs: hyperparameters associated with transform_fn. E.g., alpha and beta for ZipIt!
        Returns: A dictionay for transform operations to be performed at every point defined by PREFIX and POSTFIX, 
        on all graph models.
        """
        start_time = time()
        self.merges = {}
        self.unmerges = {}
        
        nodes = list(self.metrics.keys())
        nodes.sort()

        for node in tqdm(nodes, desc="Computing transformations: "):
            if self.start_at is None or node >= self.start_at:
                metric = self.metrics[node]
                # Maybe merge differently 
                info = self.graphs[0].get_node_info(node)
                if info['special_merge'] is not None:
                    merge, unmerge = get_merging_fn(info['special_merge'])(metric, reduce_ratio, **kwargs)
                else:
                    merge, unmerge = transform_fn(metric, reduce_ratio, **kwargs)
                
                # TODO: check if better way to do hack
                merge = merge * len(self.graphs) # Hack to deal with things not merged
                
                self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)
                
                if self.stop_at is not None and node == self.stop_at:
                    break
        
        self.compute_transform_time = time() - start_time
        return self.merges, self.unmerges
    
    def apply_transformations(self):
        """
        Applys transformations found by compute_transformations from start_at up to stop_at graph node location 
        on all graph models. 
        """
        for node in self.merges:
            merges = self.merges[node]
            unmerges = self.unmerges[node]
            for merge, unmerge, graph in zip(merges, unmerges, self.graphs):
                merger = MergeHandler(graph, merge, unmerge)
                merger.prop_back(node)

            
    def transform(self, model,
                  dataloader,
                  metric_classes=(CovarianceMetric, MeanMetric),
                  transform_fn=match_tensors_zipit,
                  prune_threshold=0.,
                  stop_at=None,
                  start_at=None,
                  **transform_kwargs
                  ):
        """ Note: this consumes the models given to the graphs. Do not modify the models you give this. """
        
        self.stop_at = stop_at
        self.start_at = start_at
        self.merged_model = model.to(self.device)
        
        if not isinstance(metric_classes, dict):
            metric_classes = { x.name: x for x in metric_classes }
        
        self.metric_classes = metric_classes
        self.transform_fn = transform_fn
        self.prune_threshold = prune_threshold
        
        self.compute_metrics(dataloader, metric_classes=metric_classes)
        self.compute_transformations(transform_fn,
                                    reduce_ratio=1 - 1. / len(self.graphs),
                                    prune_threshold=prune_threshold,
                                    **transform_kwargs
                                    )
        self.apply_transformations()
        
        self.merged_model.load_state_dict(self.get_merged_state_dict(), strict=False)
        
        self.add_hooks()
    
    def add_hooks(self):
        """ Add hooks at zip start or stop at locations for merged model and base models. """
        # Remove the hooks from the models to add or own
        self.clear_hooks()
        
        if self.start_at is not None:
            self.start_at_models = [deepcopy(g.model) for g in self.graphs]
            self.add_merge_hooks(self.merged_model, self.start_at_models, self.start_at_ptr)


    def add_merge_hooks(self, merged_model, start_at_models, start_at_ptr):
        """ Finds the first weight module that was merged but not unmerged. """
        graph = self.graphs[0]
        tmp_dict = {}

        nodes = [node for node in graph.G if
                 node not in graph.unmerged
                 and node in graph.merged
                 and self.has_weight_matrix(node)]

        for idx, node in enumerate(nodes):
            self.add_prop_hook(merged_model, node, pre=False, stop=False, loc=start_at_ptr, loc_idx=idx)
            for model in start_at_models:
                self.add_prop_hook(model, node, pre=False, stop=True, loc_idx=idx, tmp_dict=tmp_dict, tmp_dict_size=len(nodes))


    def add_prop_hook(self, model, node, pre=False, stop=False, loc=None, loc_idx=0, tmp_dict=None, tmp_dict_size=1):
        """
        Helper used for partial zipping. Add forward propogation hooks to grab intermediate outputs wherever partial zipping starts/stops. 
        These iintermediate outputs of each base model/merged model respectively will then be passed to the merged model/base models 
        respectivelty.
        """
        info = self.graphs[0].get_node_info(node)
        module = dict(model.named_modules())[info['layer']]

        def process(x):
            if len(tmp_dict) >= tmp_dict_size:
                tmp_dict.clear()
            tmp_dict[loc_idx] = x

            if len(tmp_dict) >= tmp_dict_size:
                raise MergedModelStop(tmp_dict)

            return None

        def posthook(m, x, y):
            if stop:
                return process(y)
            else:
                return loc[loc_idx]

        self.hooks.append(module.register_forward_hook(posthook))

    def forward(self, x, cat_dim=None, start_idx=None):
        """ Evaluate the combined model. """
        if self.start_at is not None:
            start_val = defaultdict(lambda: 0)
            total = 0

            for idx, model in enumerate(self.start_at_models):
                if start_idx is not None and idx != start_idx:
                    continue

                try:
                    model(x)
                except MergedModelStop as e:
                    for k, v in e.x.items():
                        start_val[k] = start_val[k] + v
                    total += 1
            
            self.start_at_ptr.clear()
            for k, v in start_val.items():
                self.start_at_ptr[k] = v / total / len(self.graphs)
            x = x[0, None].detach()
        
        try:
            # print("shape of x", x.shape)
            out = self.merged_model(x)
            # print("shape of out", out.shape)
            # print(out)
            return out
        except MergedModelStop as e:
            self.stop_at_ptr[0] = e.x[0]

            dummy_x = x[0, None].detach()
            out = []
            for idx, model in enumerate(self.head_models):
                out.append(model(dummy_x))

            self.stop_at_ptr[0] = None
            
            if cat_dim is not None:
                out = torch.cat(out, dim=cat_dim)
            
            return out
