# Brain_Cognition_Vision_Transformer
通过人类眼动数据对vit的注意力进行优化, Using human eye movement data to optimize vit's attention.

## 写在开头
所有内容都可编辑，尤其包括这个README文档，也欢迎大家来找bug
使用git进行多人协作，需要编辑的话先clone项目（不要download zip），之后每次写代码前都先pull拉取之后再写，写之后直接merge到main（暂定如此），也可以自己建立分支，写完后pull request，由其他人review之后再merge到main。（如果这句话看不懂可以群里询问）

## 开发规则（开发规则也欢迎编辑）
1. 变量/文件名使用小写字母+下划线（例如train_dataset），函数使用驼峰命名法（例如MultiHeadAttention）
2. 注释越详细越好，尽量使用英文
3. 鼓励创建多个文件夹实现不同功能，例如创立个train文件夹保存训练不同数据集的train.py代码
4. .gitignore文件中添加不需要上传的文件，不会写让chat帮你
5. 每次完成开发别忘了push到github，也支持每完成一部分就push一下
6. 模型的类需要足够的可定义变量，例如激活层、注意力头数、dropout等，方便调用