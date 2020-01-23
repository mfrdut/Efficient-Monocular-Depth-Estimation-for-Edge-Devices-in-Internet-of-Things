# coding=utf-8
import tvm
from tvm import autotvm

import numpy as np
import tvm.contrib.graph_runtime as graph_runtime


data_shape = (1, 3, 224, 224)

# load the module back.

loaded_lib = tvm.module.load('deploy_lib.tar')
#dev_lib = tvm.module.load("deploy_cuda.ptx")
#loaded_lib.import_module(dev_lib)

loaded_graph = open("deploy_graph.json").read()

loaded_params = bytearray(open("deploy_param.params", "rb").read())

cuda = True   
ctx = tvm.gpu(0) if cuda else tvm.cpu(0)

print("=> [TVM on TX2] creating TVM runtime module")
fcreate = tvm.get_global_func("tvm.graph_runtime.create")

gmodule = fcreate(loaded_graph, loaded_lib, ctx.device_type, ctx.device_id)

set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]

print("=> [TVM on TX2] feeding inputs and params into TVM module")
x = np.ones([1,3,224,224])
set_input('0', tvm.nd.array(x.astype('float32')))
gmodule["load_params"](loaded_params)

print("=> [TVM on TX2] running TVM module, saving output")
run() # not gmodule.run()

out_shape = (1, 1, 224, 224)
out = tvm.nd.empty(out_shape, "float32")
get_output(0, out)
print(out.asnumpy().shape)
#np.save(output_fp, out.asnumpy())

warmup_trials = 10
run_trials = 100
print("=> [TVM on TX2] benchmarking: {} warmup, {} run trials".format(warmup_trials, run_trials))
# run model several times as a warmup
for i in range(warmup_trials):
    run()
    ctx.sync()

# profile runtime using TVM time evaluator
ftimer = gmodule.time_evaluator("run", ctx, number=1, repeat=run_trials)
profile_result = ftimer()
profiled_runtime = profile_result[0]
print("=> [TVM on TX2] profiled runtime (in ms): {:.5f}".format(1000*profiled_runtime))

"""
# data
input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.run(data=input_data)
out_deploy = module.get_output(0).asnumpy()

# Print first 10 elements of output
print(out_deploy.shape)

  File "tune_run.py", line 21, in <module>
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
  File "/media/sda1/2019_10/tvm/python/tvm/contrib/graph_runtime.py", line 44, in create
    return GraphModule(fcreate(graph_json_str, libmod, device_type, device_id), ctx)
  File "tvm/_ffi/_cython/./function.pxi", line 286, in tvm._ffi._cy3.core.FunctionBase.__call__
  File "tvm/_ffi/_cython/./function.pxi", line 231, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 143, in tvm._ffi._cy3.core.CALL
tvm._ffi.base.TVMError: [21:25:39] /media/sda1/2019_10/tvm/src/runtime/graph/graph_runtime.cc:537: 
Check failed: pf != nullptr no such function in module: fuse_conv2d_broadcast_add_clip
"""



