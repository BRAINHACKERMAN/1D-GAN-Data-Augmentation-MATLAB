%% Synthetic Data Generation by 1-D GAN
clear;
close all;
clc;

%% 加载数据
load('data_total.mat'); % 假设数据存储在变量 Rawdata 中
Rawdata = Rawdata'; % 转置，使行是样本，列是特征
num_samples = size(Rawdata, 1); % 样本数量（1000）
num_features = size(Rawdata, 2); % 特征数量（1024）

% 生成标签向量
Target = [ones(250, 1); 2 * ones(250, 1); 3 * ones(250, 1); 4 * ones(250, 1)];

% 数据标准化（Z-score标准化）
real_data_mean = mean(Rawdata);
real_data_std = std(Rawdata);
original_data = (Rawdata - real_data_mean) ./ real_data_std;

%% 定义生成器和判别器
% 定义生成器网络（输入噪声→输出合成数据）
generatorNet = feedforwardnet([128 64]); % 两层隐藏层
generatorNet = configure(generatorNet, randn(100,1), randn(100,num_features)); % 初始化

% 定义判别器网络（输入数据→输出真伪概率）
discriminatorNet = feedforwardnet([64 32]);
discriminatorNet = configure(discriminatorNet, randn(100,num_features), randn(100,1));

% 定义生成器和判别器函数
generator = @(z) sim(generatorNet, z');  % 输入噪声z（列向量）
discriminator = @(x) sim(discriminatorNet, x'); % 输入数据x（列向量）

%% 训练GAN
% 训练参数
num_epochs = 100;     % 训练轮次
batch_size = 64;      % 批量大小
learning_rate = 0.001;% 学习率

% 训练循环
for epoch = 1:num_epochs
    for batch = 1:ceil(num_samples/batch_size)
        % 生成噪声（维度为 batch_size × 1024）
        noise = randn(batch_size, num_features);
        
        % 生成合成数据
        synthetic_data = generator(noise);
        
        % 准备真实数据批次
        real_data_batch = original_data(randperm(num_samples, batch_size), :);
        
        % 训练判别器
        d_loss_real = mean((discriminator(real_data_batch) - 1).^2); % 真实数据标签为1
        d_loss_fake = mean((discriminator(synthetic_data) - 0).^2); % 合成数据标签为0
        d_loss = (d_loss_real + d_loss_fake) / 2;
        
        % 反向传播更新判别器
        discriminatorNet = train(discriminatorNet, [real_data_batch; synthetic_data], [ones(batch_size,1); zeros(batch_size,1)]);
        
        % 训练生成器
        noise = randn(batch_size, num_features);
        synthetic_data = generator(noise);
        g_loss = mean((discriminator(synthetic_data) - 1).^2); % 欺骗判别器
        
        % 反向传播更新生成器
        generatorNet = train(generatorNet, noise, ones(batch_size,1));
    end
    
    % 打印训练信息
    fprintf('Epoch %d, Discriminator Loss: %.4f, Generator Loss: %.4f\n', epoch, d_loss, g_loss);
end

%% 生成合成数据
% 生成合成数据
noise_samples = randn(num_samples, num_features);
synthetic_data = generator(noise_samples);

% 反标准化（将数据还原到原始范围）
synthetic_data = synthetic_data .* real_data_std + real_data_mean;

%% 数据增强与分类器训练
% 划分原始数据为训练集和测试集（70%训练，30%测试）
rng(42); % 固定随机种子
split_ratio = 0.7;
[trainIdx, testIdx] = dividerand(num_samples, split_ratio, 1-split_ratio);
X_train = original_data(trainIdx, :);
y_train = Target(trainIdx);
X_test = original_data(testIdx, :);
y_test = Target(testIdx);

% 使用合成数据增强训练集
X_train_augmented = [X_train; synthetic_data];
y_train_augmented = [y_train; Target]; % 假设合成数据标签与原始数据一致

% 训练SVM分类器
Mdlsvm = fitcecoc(X_train_augmented, y_train_augmented);
pred_labels = predict(Mdlsvm, X_test);

% 计算测试精度
accuracy = sum(pred_labels == y_test) / numel(y_test) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

%% 可视化结果
% 选择两个特征进行可视化
Feature1 = 1;
Feature2 = 2;

% 原始数据分布
figure;
subplot(1,2,1);
gscatter(original_data(:,Feature1), original_data(:,Feature2), Target, 'rkgb', '.', 20);
title('Original Data Feature Distribution');

% 合成数据分布
subplot(1,2,2);
gscatter(synthetic_data(:,Feature1), synthetic_data(:,Feature2), Target, 'rkgb', '.', 20);
title('Synthetic Data Feature Distribution');