branch quant_ep_zxx_0611: 0611 merge official and yuan, update quant eval + test scripts
branch parallel_zxx: old version without updated official and yuan, self support NPU DeepEP dLLM graph

NOTE: 当我想把自己实现NPU DeepEP Graph 进行cherry-pick到quant_ep_zxx_0611时发现，有两个重要文件在新的codebase下被删除了：
CONFLICT (modify/delete): python/sglang/srt/model_executor/cuda_graph_runner.py deleted in HEAD and modified in 20ce28b2e (Support NPU DeepEP dLLM graph).  Version 20ce28b2e (Support NPU DeepEP dLLM graph) of python/sglang/srt/model_executor/cuda_graph_runner.py left in tree.
CONFLICT (modify/delete): python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py deleted in HEAD and modified in 20ce28b2e (Support NPU DeepEP dLLM graph).  Version 20ce28b2e (Support NPU DeepEP dLLM graph) of python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py left in tree.

另一个比较大的改动是：python/sglang/srt/server_args.py 中的 _handle_multi_item_scoring()。我之前parallel_zxx中的 _handle_piecewise_cuda_graph () 这个func完全删除了。然后 _handle_a2a_moe (self)也改动很大，我也注释了。

我感觉首先要做的是验证新的codebase能否EP执行