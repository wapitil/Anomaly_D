import onnx

m = onnx.load("runs/res_640/onnx/res_640.onnx")
print("ir_version:", m.ir_version)
print("opset:", [(o.domain, o.version) for o in m.opset_import])
