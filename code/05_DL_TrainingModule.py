import numpy as np
from paddle.io import Dataset
import pandas as pd
from sklearn import preprocessing

# 数据集定义
class CancerNormalDataset(Dataset):
    def __init__(self, data, mode):
        data = data[data['model'] == mode]
        self.data_normal = data.iloc[:,:-2].values
        self.label=data.iloc[:,2625].values
        self.name = data.index

    def __getitem__(self, idx):
        item = np.array(self.data_normal[idx],dtype='float32')
        label = int(self.label[idx])
        return item, np.array(label, dtype='int64')
    
    def Mygetitem(self, idx):
        item = np.array(self.data_normal[idx],dtype='float32')
        label = int(self.label[idx])
        name = self.name[idx]
        return item, np.array(label, dtype='int64'),name

    def __len__(self):
        return len(self.data_normal)
    

# Load the dataset, setting the first column as the index
data = pd.read_csv('../data/dataset.csv', index_col=0)
# Drop rows with missing values
data = data.dropna()

#labelEncoder编码处理数据
le = preprocessing.LabelEncoder()
le.fit(data['kind'].values.tolist())
print(le.classes_)
le.transform(data['kind'].values.tolist())

data['kind']  = le.transform(data['kind'].values.tolist())
data['kind'] = 1-data['kind']

train_dataset=CancerNormalDataset(data, mode='train')
test_dataset=CancerNormalDataset(data, mode='test')

print('features dim:'+str(len(train_dataset[0][0])))
print(train_dataset[1])

import paddle
import paddle.nn.functional as F
from paddle.nn import Linear

# 定义动态图
class MyDNN(paddle.nn.Layer):
    def __init__(self):
        super(MyDNN, self).__init__()
        self.fc1 = Linear(2625,1000)
        self.fc2 = Linear(1000,200)
        self.fc3= Linear(200,50)
        self.fc4= Linear(50,2)
    
    # 网络的前向计算函数
    def forward(self, inputs):
        x = self.fc1(inputs)
        x=F.relu(x)
        x = self.fc2(x)
        x=F.sigmoid(x) 
        x = self.fc3(x)
        x=F.sigmoid(x)
        x = self.fc4(x)
        pred=F.softmax(x)
        return pred
    
model = MyDNN()  # 网络实例化
paddle.summary(model,(1,2625))  # 网络结构查看

model = paddle.Model(MyDNN())
# 定义损失函数
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 训练可视化VisualDL工具的回调函数
visualdl = paddle.callbacks.VisualDL(log_dir='visualdl_log')

# 启动模型全流程训练
model.fit(train_dataset,            # 训练数据集
          eval_data=test_dataset,
          epochs=100,            # 总的训练轮次
          batch_size = 64,    # 批次计算的样本量大小
          shuffle=True,             # 是否打乱样本集
          verbose=1,              # 日志展示格式
          callbacks=[visualdl]
          )     # 回调函数使用
# 保存模型
save_dir='./chk_points/', # 分阶段的训练模型存储路径
model.save('model_save_dir')

result = model.evaluate(test_dataset, verbose=1)

from paddle.io import DataLoader, BatchSampler  
# 创建一个空的列表来存储预测结果  
name =[]
predictions = []  
scores=[]

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print()
# 逐一预测测试集数据  
with paddle.no_grad():  # 不计算梯度，节省计算资源  
    for batch_id, (data, _) in enumerate(test_loader()):  
        # 获取单个样本的特征和标签（如果需要的话）  
        # 注意：这里我们假设test_loader每次只返回一个样本  
        features = data.numpy()  
        features = paddle.to_tensor(features)  
        # 进行预测  
        outputs = model.predict(features)  
        for o in outputs[0]:
            predicted = np.argmax(o) 
            o=o.tolist()
            predictions.append(predicted) 
            score = o[predicted] if predicted==1 else 1 - o[predicted]
            scores.append(score)

for i in range(len(predictions)):
    name.append(test_dataset.Mygetitem(i)[2])

# 将预测结果和索引整理成DataFrame  
df = pd.DataFrame({  
    'name': name,  # 测试集样本的索引  
    'score':scores,
    'prediction': predictions  # 预测结果  
})  
  
# 如果需要，可以将DataFrame保存到CSV文件  
df.to_csv('../res/dl_res.csv', index=False)
  
# 打印预测结果  
print(df)