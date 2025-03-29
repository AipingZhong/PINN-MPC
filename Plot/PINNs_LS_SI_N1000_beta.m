%% Global parameter settings
clear;
clc;
S_star = 1/3;          % Susceptible threshold
I_max = 0.10;          % Infection threshold
u_max = 0.4;           % Maximum control input
t_sim = 50;            % Total simulation time

% Visualization parameters
font_size = 26;        % Uniform font size
legend_size = 16;      % Uniform legend font size
threshold_font_size = 32; % Font size for threshold labels
line_width = 4.5;      % Main line width
threshold_line_width = 6; % Threshold line width
area_alpha = 0.9;      % Transparency for SIR confidence interval (higher = darker)
area_alpha2 = 0.6;     % Transparency for beta confidence interval (higher = darker)
marker_size = 6;       % Marker size for observations
marker_line_width = 2; % Marker line width

% Professional color scheme
colors = struct(...
    'LS_true',    [0.40, 0.70, 0.90],...   % LS true values (blue)
    'LS_mean',    [0.00, 0.40, 0.80],...   % LS mean values
    'LS_ci',      [0.85, 0.92, 0.98],...   % LS confidence interval
    'LS_obs',     [0.20, 0.60, 1.00],...   % LS observations
    'SI_true',    [0.40, 0.80, 0.40],...   % SI true values (green)
    'SI_mean',    [0.20, 0.60, 0.20],...   % SI mean values
    'SI_ci',      [0.85, 0.95, 0.85],...   % SI confidence interval
    'SI_obs',     [0.20, 0.40, 0.20],...   % SI observations
    'PINN_true',  [0.90, 0.50, 0.10],...   % PINN true values (orange)
    'PINN_mean',  [0.80, 0.40, 0.00],...   % PINN mean values
    'PINN_ci',    [0.98, 0.85, 0.70],...   % PINN confidence interval
    'PINN_obs',   [0.70, 0.30, 0.00],...   % PINN observations
    'true_beta',  [0.65, 0.25, 0.45],...   % True beta (dark magenta)
    'threshold',  [0.2, 0.2, 0.2]);        % Threshold line color (dark gray)

%% Data loading
data_csv1 = readtable('PINNs_kappa0.01/mean_std.csv'); % PINNs
data_csv2 = readtable('LS-PINNs_kappa0.01/mean_std.csv'); % LS-PINNs
data_csv3 = readtable('SI-PINNs_kappa0.01/mean_std2.csv'); % SI-PINNs

%% SIR titled subplots
states = {'S', 'I', 'R'};
ylabels = {'$S$', '$I$', '$R$'};

for i = 1:3
    fig = figure('Units','normalized','Position',[0.1,0.1,0.8,0.8]); % left bottom width height
    
    % ================== Subplot 1: PINNs (data_csv1) ==================
    subplot(3,1,1)
    hold on;
    
    % Confidence interval
    fill([data_csv1.Time; flipud(data_csv1.Time)],...
         [data_csv1.(sprintf('%s_mean',states{i})) + data_csv1.(sprintf('%s_std',states{i}));...
          flipud(data_csv1.(sprintf('%s_mean',states{i})) - data_csv1.(sprintf('%s_std',states{i})))],...
         colors.PINN_ci, 'EdgeColor','none', 'FaceAlpha',area_alpha);
    
    % True values
    h1 = plot(data_csv1.Time, data_csv1.(sprintf('%s_true',states{i})),...
        'Color',colors.PINN_true, 'LineWidth',line_width);
    
    % Estimated values
    h2 = plot(data_csv1.Time, data_csv1.(sprintf('%s_mean',states{i})), '-.',...
        'Color',colors.PINN_mean, 'LineWidth',line_width);
    
    % Observations for I only
    if i == 2
        h3 = plot(data_csv1.Time, data_csv1.I_observation, 'x',...
            'Color',colors.PINN_obs, 'MarkerSize',marker_size,...
            'LineWidth',marker_line_width, 'MarkerFaceColor','none');
    end
    
    % Threshold line and label
    if i == 1
        yline(S_star, '--', 'Color',colors.threshold,...
              'LineWidth',threshold_line_width);
        text(t_sim*0.075, S_star, '$S^\star$', 'VerticalAlignment','bottom',...
             'HorizontalAlignment','right', 'FontSize',font_size,...
             'Color',colors.threshold, 'Interpreter', 'latex');
    elseif i == 2
        yline(I_max, '--', 'Color',colors.threshold,...
              'LineWidth',threshold_line_width);
        text(t_sim*0.1, I_max, '$I_{max}$', 'VerticalAlignment','bottom',...
             'HorizontalAlignment','right', 'FontSize',font_size,...
             'Color',colors.threshold, 'Interpreter', 'latex');
    end
    
    % Axis settings
    xlim([0 t_sim]);
    if i == 2
        ylim([0 0.18]);
    else
        ylim([0 1]);
    end
    set(gca, 'FontSize',font_size, 'LineWidth',1.5, 'GridAlpha',0.25);
    ylabel(ylabels{i}, 'FontSize',font_size+2, 'Interpreter', 'latex');
    title('PINNs', 'FontSize',font_size+2);
    grid on;
    box on;
    
    % Legend
    if i == 2
        legend([h1, h2, h3], {'True','Est.','Obs.'},...
               'Location','best', 'FontSize',legend_size,...
               'Interpreter','latex','NumColumns', 1);
    else
        legend([h1, h2], {'True','Est.'},...
               'Location','best', 'FontSize',legend_size,...
               'Interpreter','latex');
    end
    
    % ================== Subplot 2: LS-PINNs (data_csv2) ==================
    subplot(3,1,2)
    hold on;

    % Confidence interval
    fill([data_csv2.Time; flipud(data_csv2.Time)],...
         [data_csv2.(sprintf('%s_mean',states{i})) + data_csv2.(sprintf('%s_std',states{i}));...
          flipud(data_csv2.(sprintf('%s_mean',states{i})) - data_csv2.(sprintf('%s_std',states{i})))],...
         colors.LS_ci, 'EdgeColor','none', 'FaceAlpha',area_alpha);
    
    % True values
    h4 = plot(data_csv2.Time, data_csv2.(sprintf('%s_true',states{i})),...
        'Color',colors.LS_true, 'LineWidth',line_width);
    
    % Estimated values
    h5 = plot(data_csv2.Time, data_csv2.(sprintf('%s_mean',states{i})), '-.',...
        'Color',colors.LS_mean, 'LineWidth',line_width);
    
    % Observations for I only
    if i == 2
        h6 = plot(data_csv2.Time, data_csv2.I_observation, 'o',...
            'Color',colors.LS_obs, 'MarkerSize',marker_size,...
            'LineWidth',marker_line_width);
    end
    
    % Threshold line and label
    if i == 1
        yline(S_star, '--', 'Color',colors.threshold,...
              'LineWidth',threshold_line_width);
        text(t_sim*0.075, S_star, '$S^\star$', 'VerticalAlignment','bottom',...
             'HorizontalAlignment','right', 'FontSize',font_size,...
             'Color',colors.threshold, 'Interpreter', 'latex');
    elseif i == 2
        yline(I_max, '--', 'Color',colors.threshold,...
              'LineWidth',threshold_line_width);
        text(t_sim*0.1, I_max, '$I_{max}$', 'VerticalAlignment','bottom',...
             'HorizontalAlignment','right', 'FontSize',font_size,...
             'Color',colors.threshold, 'Interpreter', 'latex');
    end
    
    % Axis settings
    xlim([0 t_sim]);
    if i == 2
        ylim([0 0.18]);
    else
        ylim([0 1]);
    end
    set(gca, 'FontSize',font_size, 'LineWidth',1.5, 'GridAlpha',0.25);
    ylabel(ylabels{i}, 'FontSize',font_size+2, 'Interpreter', 'latex');
    title('LS-PINNs', 'FontSize',font_size+2);
    grid on;
    box on;
    
    % Legend
    if i == 2
        legend([h4, h5, h6], {'True','Est.','Obs.'},...
               'Location','best', 'FontSize',legend_size,...
               'Interpreter','latex','NumColumns', 1);
    else
        legend([h4, h5], {'True','Est.'},...
               'Location','best', 'FontSize',legend_size,...
               'Interpreter','latex');
    end
    
    % ================== Subplot 3: SI-PINNs (data_csv3) ==================
    subplot(3,1,3)
    hold on;

    % Confidence interval
    fill([data_csv3.Time; flipud(data_csv3.Time)],...
         [data_csv3.(sprintf('%s_mean',states{i})) + data_csv3.(sprintf('%s_std',states{i}));...
          flipud(data_csv3.(sprintf('%s_mean',states{i})) - data_csv3.(sprintf('%s_std',states{i})))],...
         colors.SI_ci, 'EdgeColor','none', 'FaceAlpha',area_alpha);
    
    % True values
    h7 = plot(data_csv3.Time, data_csv3.(sprintf('%s_true',states{i})),...
        'Color',colors.SI_true, 'LineWidth',line_width);
    
    % Estimated values
    h8 = plot(data_csv3.Time, data_csv3.(sprintf('%s_mean',states{i})), '-.',...
        'Color',colors.SI_mean, 'LineWidth',line_width);
    
    % Observations for I only
    if i == 2
        h9 = plot(data_csv3.Time, data_csv3.I_observation, '*',...
            'Color',colors.SI_obs, 'MarkerSize',marker_size,...
            'LineWidth',marker_line_width, 'MarkerFaceColor','none');
    end
    
    % Threshold line and label
    if i == 1
        yline(S_star, '--', 'Color',colors.threshold,...
              'LineWidth',threshold_line_width);
        text(t_sim*0.075, S_star, '$S^\star$', 'VerticalAlignment','bottom',...
             'HorizontalAlignment','right', 'FontSize',font_size,...
             'Color',colors.threshold, 'Interpreter', 'latex');
    elseif i == 2
        yline(I_max, '--', 'Color',colors.threshold,...
              'LineWidth',threshold_line_width);
        text(t_sim*0.1, I_max, '$I_{max}$', 'VerticalAlignment','bottom',...
             'HorizontalAlignment','right', 'FontSize',font_size,...
             'Color',colors.threshold, 'Interpreter', 'latex');
    end
    
    % Axis settings
    xlim([0 t_sim]);
    if i == 2
        ylim([0 0.18]);
    else
        ylim([0 1]);
    end
    set(gca, 'FontSize',font_size, 'LineWidth',1.5, 'GridAlpha',0.25);
    xlabel('Time (days)', 'FontSize',font_size+2);
    ylabel(ylabels{i}, 'FontSize',font_size+2, 'Interpreter', 'latex');
    title('SI-PINNs', 'FontSize',font_size+2);
    grid on;
    box on;
    
    % Legend
    if i == 2
        legend([h7, h8, h9], {'True','Est.','Obs.'},...
               'Location','best', 'FontSize',legend_size,...
               'Interpreter','latex','NumColumns', 1);
    else
        legend([h7, h8], {'True','Est.'},...
               'Location','best', 'FontSize',legend_size,...
               'Interpreter','latex');
    end
end

%% Control input and beta
figure('Units','normalized','Position',[0.1,0.1,0.8,0.8]);

% ========== Subplot 1: Control input comparison ==========
subplot(3,1,1)
hold on;

% PINNs control
stairs(data_csv1.Time, data_csv1.u_actual,...
    'Color',colors.PINN_mean, 'LineWidth',line_width, 'DisplayName','PINNs');

% LS-PINNs control
stairs(data_csv2.Time, data_csv2.u_actual,...
    'Color',colors.LS_mean, 'LineWidth',line_width, 'DisplayName','LS-PINNs');

% SI-PINNs control
stairs(data_csv3.Time, data_csv3.u_actual,...
    'Color',colors.SI_mean, 'LineWidth',line_width, 'DisplayName','SI-PINNs');

% Max control threshold and label
yline(u_max, '--', 'Color',colors.threshold,...
       'LineWidth',threshold_line_width, 'HandleVisibility','off');
text(t_sim*0.125, u_max, '$u_{max}$', 'VerticalAlignment','bottom',...
     'HorizontalAlignment','right', 'FontSize',threshold_font_size,...
     'Color',colors.threshold, 'Interpreter', 'latex');

% Formatting
xlim([0 t_sim]);
ylim([0 0.60]);
set(gca, 'FontSize',font_size, 'LineWidth',1.5, 'GridAlpha',0.25);
xlabel('Time (days)', 'FontSize',font_size);
ylabel('MPC $u$', 'FontSize',font_size, 'Interpreter', 'latex');
legend('Location','northeast', 'FontSize',legend_size, 'NumColumns',2);
grid on;
box on;
hold off;

% ========== Subplot 2: Beta parameter comparison ==========
subplot(3,1,2)
hold on;

% True curve
p1 = plot(data_csv1.Time, data_csv1.beta_true, '-',...
        'Color',colors.true_beta, 'LineWidth',line_width);

% Estimated curves
p2 = plot(data_csv1.Time, data_csv1.beta_mean, '--',...
        'Color',colors.PINN_mean, 'LineWidth',line_width);
p3 = plot(data_csv2.Time, data_csv2.beta_mean, '--',...
        'Color',colors.LS_mean, 'LineWidth',line_width);
p4 = plot(data_csv3.Time, data_csv3.beta_mean, '--',...
        'Color',colors.SI_mean, 'LineWidth',line_width);

% Confidence intervals
fill([data_csv1.Time; flipud(data_csv1.Time)],...
     [data_csv1.beta_mean+data_csv1.beta_std; flipud(data_csv1.beta_mean-data_csv1.beta_std)],...
     colors.PINN_ci, 'EdgeColor','none', 'FaceAlpha',area_alpha2);
fill([data_csv2.Time; flipud(data_csv2.Time)],...
     [data_csv2.beta_mean+data_csv2.beta_std; flipud(data_csv2.beta_mean-data_csv2.beta_std)],...
     colors.LS_ci, 'EdgeColor','none', 'FaceAlpha',area_alpha2);
fill([data_csv3.Time; flipud(data_csv3.Time)],...
     [data_csv3.beta_mean+data_csv3.beta_std; flipud(data_csv3.beta_mean-data_csv3.beta_std)],...
     colors.SI_ci, 'EdgeColor','none', 'FaceAlpha',area_alpha2);

% Formatting
xlim([0 t_sim]);
set(gca, 'FontSize',font_size, 'LineWidth',1.5, 'GridAlpha',0.25);
ylabel('$\beta$', 'FontSize',font_size, 'Interpreter', 'latex');
legend([p1 p2 p3 p4], {'True $\beta$','PINNs Est.','LS-PINNs Est.','SI-PINNs Est.'},...
       'Location','southwest', 'FontSize',legend_size, 'Interpreter', 'latex','NumColumns',2);
grid on;
box on;
hold off;

% ========== Subplot 3: Relative error of beta ==========
subplot(3,1,3)
hold on;

% Compute relative error
csv1_beta_error = abs(data_csv1.beta_mean - data_csv1.beta_true) ./ data_csv1.beta_true;
csv2_beta_error = abs(data_csv2.beta_mean - data_csv2.beta_true) ./ data_csv2.beta_true;
csv3_beta_error = abs(data_csv3.beta_mean - data_csv3.beta_true) ./ data_csv3.beta_true;

% Error curves
p1 = plot(data_csv1.Time, csv1_beta_error, '-',...
        'Color',colors.PINN_mean, 'LineWidth',line_width);
p2 = plot(data_csv2.Time, csv2_beta_error, '-',...
        'Color',colors.LS_mean, 'LineWidth',line_width);
p3 = plot(data_csv3.Time, csv3_beta_error, '-',...
        'Color',colors.SI_mean, 'LineWidth',line_width);

% Formatting
xlim([0 t_sim]);
set(gca, 'YScale','log', 'FontSize',font_size, 'LineWidth',1.5);
xlabel('Time (days)', 'FontSize',font_size);
ylabel('RAE of $\hat \beta$', 'FontSize',font_size, 'Interpreter', 'latex');
legend([p1 p2 p3], {'PINNs','LS-PINNs','SI-PINNs'},...
       'Location','northwest', 'FontSize',legend_size,'NumColumns',2);
grid on;
box on;
hold off;

%% rMSE for S, I, R, and β in selected range
time = data_csv1.Time;

% Set time range for evaluation
t1 = 0;  % Start time
t2 = 50;  % End time

% Dataset configuration: name, table object, display label
datasets = {
    struct('name', 'data_csv1', 'data', data_csv1, 'label', 'PINNs'), 
    struct('name', 'data_csv2', 'data', data_csv2, 'label', 'LS-PINNs'), 
    struct('name', 'data_csv3', 'data', data_csv3, 'label', 'SI-PINNs')
};

% Preallocate results
results = struct();

for d = 1:length(datasets)
    % Get current dataset
    dataset = datasets{d};
    data = dataset.data;
    
    % Filter time range
    time_filter = (data.Time >= t1) & (data.Time <= t2);
    
    % Compute errors for each state and parameter
    results(d).label = dataset.label;
    
    % S error
    S_true = data.S_true(time_filter);
    S_mean = data.S_mean(time_filter);
    results(d).S = compute_errors(S_true, S_mean);
    
    % I error
    I_true = data.I_true(time_filter);
    I_mean = data.I_mean(time_filter);
    results(d).I = compute_errors(I_true, I_mean);
    
    % R error
    R_true = data.R_true(time_filter);
    R_mean = data.R_mean(time_filter);
    results(d).R = compute_errors(R_true, R_mean);
    
    % β error
    beta_true = data.beta_true(time_filter);
    beta_mean = data.beta_mean(time_filter);
    results(d).beta = compute_errors(beta_true, beta_mean);
    
    % Global I_max across full time range
    results(d).I_max = max(data.I_true);  % Max I_true over all time
end

% Error computation function
function err = compute_errors(true_vals, mean_vals)
    err = struct(...
        'rMSE', sum((mean_vals - true_vals).^2) / sum(true_vals.^2));
end

% Unified result output
fprintf('Error analysis for time range [%d, %d]:\n', t1, t2);
for d = 1:length(results)
    fprintf('\n=== %s ===\n', results(d).label);
    fprintf('S: rMSE=%.3e  ', results(d).S.rMSE);
    fprintf('I: rMSE=%.3e  ', results(d).I.rMSE);
    fprintf('R: rMSE=%.3e  ', results(d).R.rMSE);
    fprintf('β: rMSE=%.3e  ', results(d).beta.rMSE);
    fprintf('I_max = %.4f\n', results(d).I_max);  % Output global I_max
end