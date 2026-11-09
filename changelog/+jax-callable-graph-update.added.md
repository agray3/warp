Patch CUDA graphs of `wp.jax_callable()` callables in `JaxCallableGraphMode.WARP` mode with new buffer
addresses instead of capturing new graphs. XLA can move a callable's arguments on every execution of a jitted
function, which used to cost a full graph capture per execution: for the MuJoCo Warp step graph in a JAX training
loop, a capture took about 10 ms of host time with the GPU idle, while a patch takes about 0.2 ms. Patching is on
by default and controlled by the new `graph_update` argument and `wp.config.jax_callable_graph_update` setting;
graphs that copy data from host memory are only patched with `graph_update=True`.
