%% Synthetic Data Generation by Very Basic 1-D GAN
clear;
close all;
clc;

%% 加载原始数据集：以fisheriris数据集为例，该数据集由3类，每类50个样本，每个样本4个特征，
% 共 150个样本*4个特征
load data_total.mat;
original_data=reshape(Rawdata,1,[]);                                 % 预处理-将矩阵转换成向量
Target(1:250)=1;Target(251:500)=2;Target(501:750)=3;Target(701:1000)=4;Target=Target'; % 标签
real_data_mean = mean (original_data);                            % 均值
real_data_std = std (original_data);                              % 标准差

%% 定义生成器和判别器
generator = @(z) original_data;                                   % 映射
discriminator = @(x) (x - original_data);                         % Z-score 标准化

%% 训练参数
num_samples = 1000;
num_epochs = 100;
batch_size = 64;
learning_rate = 0.01;
% 每次运行生成的样本与原始数据中的样本数量相等
% 所以，3次运行意味着原始数据* 3
Runs= 1;                                                         % 运行次数
for i=1:Runs
    % 训练循环
    for epoch = 1:num_epochs
        for batch = 1:num_samples/batch_size
            % 为生成器generator生成噪声样本
            noise = randn(batch_size, 1);
            % 使用生成器generator生成合成数据
            synthetic_data = generator(noise);
            % 训练判别器discriminator区分真实数据和合成数据
            discriminator_loss = mean((discriminator(synthetic_data) - noise).^2);
            % 更新生成器generator以欺骗判别器discriminator
            generator_loss = mean((discriminator(generator(noise)) - noise).^2);
            % 更新生成器generator和判别器discriminator参数
            generator = @(z) generator(z) - learning_rate * generator_loss;
            discriminator = @(x) discriminator(x) - learning_rate * discriminator_loss;
        end
        % 打印信息
        Run = [' Epoch "',num2str(epoch)];
        disp(Run);
    end
    % 使用经过训练的生成器generator生成合成数据
    noise_samples = randn(num_samples, 1);
    synthetic_data= generator(noise_samples);
    Syn(i,:)=synthetic_data;
    Run2 = [' Run "',num2str(Runs)];
    disp(Run2);
end

%% 将单元格转换为矩阵
S = size(Syn(Runs));
SO = size (Rawdata);
SF = SO (1,2);
SO = SO (1,1);
for i=1:Runs
    Syn2{i}=reshape(Syn(i,:),[SO,SF]);
    Syn2{i}(:,end+1)=Target;
end
Synthetic3 = cell2mat(Syn2');
SyntheticData=Synthetic3(:,1:end-1);
SyntheticLbl=Synthetic3(:,end);

%% 绘图数据
Feature1=1;
Feature2=1024;
f1=Rawdata(:,Feature1); % feature1
f2=Rawdata(:,Feature2); % feature 2
ff1=SyntheticData(:,Feature1); % feature1
ff2=SyntheticData(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
% 原始数据
subplot(3,3,1)
area(Rawdata, 'linewidth',1); title('Original Data'); 
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
% 生成数据
subplot(3,3,2)
area(SyntheticData, 'linewidth',1); title('Synthetic Data'); 
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
% 原始数据的两个特征的分布
subplot(3,3,3)
gscatter(f1,f2,Target,'rkgb','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
% 合成数据的两个特征的分布
subplot(3,3,4)
gscatter(ff1,ff2,SyntheticLbl,'rkgb','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
% 原始数据和合成数据的直方图分布
subplot(3,3,5) 
histogram(Rawdata, 'Normalization', 'probability', 'DisplayName', 'Original Data');% 原始数据的分布
hold on;
histogram(SyntheticData, 'Normalization', 'probability', 'DisplayName', 'Synthetic Data');% 合成数据的分布
legend('Original','Synthetic')
% 原始数据和合成数据的概率分布
subplot(3,3,6)
histogram(synthetic_data, 'Normalization', 'probability', 'DisplayName', 'Synthetic Data');
hold on;
x_range = linspace(real_data_mean - 3 * real_data_std, real_data_mean + 3 * real_data_std, 100);
real_data_distribution = normpdf(x_range, real_data_mean, real_data_std);
plot(x_range, real_data_distribution, 'r', 'LineWidth', 2, 'DisplayName', 'Real Data Distribution');
legend();
xlabel('Value');
ylabel('Probability');
title('Real Data vs. Synthetic Data Distribution');
% 原始数据的四个特征的箱线图
subplot(3,3,7)
boxplot(Rawdata);title('Original');
% 合成数据的四个特征的箱线图
subplot(3,3,8)
boxplot(SyntheticData);title('Synthetic');

subplot(3,3,9)
probplot(Rawdata);title('Original');
hold on;
probplot(SyntheticData);title('Original and Synthetic');

%% 训练和测试分类器-以SVM为例
% 利用合成数据训练
Mdlsvm  = fitcecoc(SyntheticData,SyntheticLbl);
CVMdlsvm = crossval(Mdlsvm);
SVMError = kfoldLoss(CVMdlsvm);
SVMAccAugTrain = (1 - SVMError)*100;
% 预测新样本(整个原始数据集)
[label5,score5,cost5] = predict(Mdlsvm,Rawdata);
% 测试误差和精度计算
sizlbl=size(Target); sizlbl=sizlbl(1,1);
countersvm=0; %
misindexsvm=0; % 误分类索引
for i=1:sizlbl
    if Target(i)~=label5(i)
        misindex(i)=i;
        countersvm=countersvm+1;
    end
end
% 测试精度
TestErrAugsvm = countersvm*100/sizlbl; 
SVMAccAugTest = 100 - TestErrAugsvm;
% SVM的结果
AugResSVM = [' Synthetic Train SVM "',num2str(SVMAccAugTrain),'" Test on Original Dataset"', num2str(SVMAccAugTest),'"'];
disp(AugResSVM);
