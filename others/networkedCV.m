for ii = 1:numel(baseline_MAE)
    BASELINE_mean(ii) = mean(baseline_MAE(1:ii));
    Networked_mean(ii) = mean(networked_MAE(1:ii));
    single_mean(ii) = mean(single_MAE(1:ii));
    SCAFFOLD_mean(ii) = mean(SCAFFOLD_MAE(1:ii));
end


plot(BASELINE_mean,'DisplayName','FedAvg');
hold on;
plot(Networked_mean,'DisplayName','Networked CV');
plot(single_mean,'DisplayName','Single CV');
plot(SCAFFOLD_mean,'DisplayName','SCAFFOLD');
hold off;
