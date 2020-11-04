import os
from tensorflow.python import pywrap_tensorflow

model_dir = os.getcwd() # 获取当前文件工作路径
print(model_dir)#输出当前工作路径
checkpoint_path = model_dir + "\\models_32x32\\ing\\model_32x32_700.dat"

print(checkpoint_path)#输出读取的文件路径
# 从checkpoint文件中读取参数
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# 输出变量名称及变量值
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))