class AttentionMapExtractor:
    def __init__(self):
        self.attn_maps = []

    def hook_fn(self, module, input, output):
        # output is output of decoder layer，not icluding attention map，but input[1] is memory
        # get from module.multihead_attn
        q = module.norm2(input[0]) if module.norm_first else input[0]  # tgt
        k = input[1]  # memory
        attn_output, attn_weights = module.multihead_attn(
            query=q,
            key=k,
            value=k,
            need_weights=True,
            average_attn_weights=False  # save heads
        )
        self.attn_maps.append(attn_weights.detach().cpu())

    def register_hooks(self, transformer_decoder):
        for layer in transformer_decoder.layers:
            layer.register_forward_hook(self.hook_fn)

    def get_map(self, layer_idx=0, batch_idx=0, head_idx=0):
        return self.attn_maps[layer_idx][batch_idx][head_idx]

    def clear(self):
        self.attn_maps.clear()

    def visualize(self, layer_idx=0, batch_idx=0, head_idx=0, title="Cross Attention"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        attn = self.get_map(layer_idx, batch_idx, head_idx)
        plt.figure(figsize=(8, 6))
        sns.heatmap(attn, cmap='viridis')
        plt.title(title)
        plt.xlabel("Memory (KV)")
        plt.ylabel("Query (TGT)")
        plt.show()
