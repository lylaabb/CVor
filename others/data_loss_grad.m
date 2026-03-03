grad(grad==0)=[];
grad1(grad1==0)=[];
grad2(grad2==0)=[];

loss(loss==0)=[];
loss1(loss1==0)=[];
loss2(loss2==0)=[];

for ii = 1:numel(loss)
    loss_mean(ii) = mean(loss(1:ii));
end
for ii = 1:numel(loss1)
    loss1_mean(ii) = mean(loss1(1:ii));
end
for ii = 1:numel(loss2)
    loss2_mean(ii) = mean(loss2(1:ii));
end

for ii = 1:numel(grad)
    grad_mean(ii) = mean(grad(1:ii));
end
for ii = 1:numel(grad1)
    grad1_mean(ii) = mean(grad1(1:ii));
end
for ii = 1:numel(grad2)
    grad2_mean(ii) = mean(grad2(1:ii));
end

for ii = 1:numel(reward)
    reward_mean(ii) = mean(reward(1:ii));
end
for ii = 1:numel(reward1)
    reward1_mean(ii) = mean(reward1(1:ii));
end
for ii = 1:numel(reward2)
    reward2_mean(ii) = mean(reward2(1:ii));
end

figure
plot(grad_mean)
hold on 
plot(grad1_mean)
plot(grad2_mean)

figure
plot(loss_mean)
hold on 
plot(loss1_mean)
plot(loss2_mean)

figure
plot(reward_mean)
hold on 
plot(reward1_mean)
plot(reward2_mean)
