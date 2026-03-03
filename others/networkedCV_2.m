for ii = 1:numel(networked_MAE)
    BASELINE_mean(ii) = mean(baseline_MAE(1:ii));
    Networked_mean(ii) = mean(networked_MSE(1:ii));
    SCAFFOLD_mean(ii) = mean(SCAFFOLD_MSE(1:ii));
end


plot(Networked_mean,'DisplayName','Networked CV');
hold on;
plot(BASELINE_mean,'DisplayName','FedAvg');
plot(SCAFFOLD_mean,'DisplayName','SCAFFOLD');
hold off;
