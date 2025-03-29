%% Global parameter settings
clear;
clc;
S_star = 1/3;          % Susceptible threshold
I_max = 0.1;           % Infection threshold
u_max = 0.4;           % Maximum control
t_sim = 50;            % Total simulation time

% Visualization parameters
font_size = 26;        % Uniform font size
legend_size = 16;      % Uniform legend font size
threshold_font_size = 32; % Font size for threshold annotations
line_width = 4.5;        % Main line width
threshold_line_width = 6;  % Threshold line width
area_alpha = 0.9;      % Transparency for SIR confidence interval (higher = darker)
area_alpha2 = 0.6;     % Transparency for beta confidence interval (higher = darker)
marker_size = 6;       % Marker size for observations
marker_line_width = 2;  % Marker line width

% Professional color scheme
colors = struct(...
    'LS_true',    [0.40, 0.70, 0.90],...   % LS true values (blue)
    'LS_mean',    [0.00, 0.40, 0.80],...   % LS mean values
    'LS_ci',      [0.85, 0.92, 0.98],...   % LS confidence interval
    'LS_obs',     [0.20, 0.60, 1.00],...   % LS observations
    'SI_true', [0.40, 0.80, 0.40],...      % SI true values (green)
    'SI_mean', [0.20, 0.60, 0.20],...      % SI mean values
    'SI_ci',   [0.85, 0.95, 0.85],...      % SI confidence interval
    'SI_obs', [0.20, 0.40, 0.20],...       % SI observations (darker green)
    'true_beta', [0.65, 0.25, 0.45],...    % True beta (dark purple-red)
    'threshold',  [0.2, 0.2, 0.2]);        % Threshold line color (dark gray)

%% Load data
data_csv1 = readtable('LS-PINNs_kappa1/mean_std.csv'); % LS
data_csv2 = readtable('SI-PINNs_kappa1/mean_std.csv'); % SI

%% SIR states and control visualization (2x2 subplots)
states = {'S', 'I', 'R'};
ylabels = {'$S$', '$I$', '$R$', 'MPC $u$'};
figure('Units','normalized','Position',[0.1,0.1,0.8,0.8]);

% Store each subplot handle
subplot_handles = gobjects(4,1);

for i = 1:4
    subplot_handles(i) = subplot(2,2,i);
    hold on;
    
    if i <= 3  % First three plots (S, I, R)
        % ================== Initialize legend handles ==================
        h = gobjects(6,1);
        
        % ================== LS-PINNs plotting ==================
        fill([data_csv1.Time; flipud(data_csv1.Time)],...
             [data_csv1.(sprintf('%s_mean',states{i})) + data_csv1.(sprintf('%s_std',states{i}));...
              flipud(data_csv1.(sprintf('%s_mean',states{i})) - data_csv1.(sprintf('%s_std',states{i})))],...
             colors.LS_ci, 'EdgeColor','none', 'FaceAlpha',area_alpha);
        
        h(1) = plot(data_csv1.Time, data_csv1.(sprintf('%s_true',states{i})),...
            'Color',colors.LS_true, 'LineWidth',line_width);
        
        h(2) = plot(data_csv1.Time, data_csv1.(sprintf('%s_mean',states{i})), '-.',...
            'Color',colors.LS_mean, 'LineWidth',line_width);
        
        if i == 2  % Observations for I
            h(3) = plot(data_csv1.Time, data_csv1.I_observation, 'o',...
                'Color',colors.LS_obs, 'MarkerSize',marker_size,...
                'LineWidth',marker_line_width);
        end
        
        % ================== PINNs plotting ==================
        fill([data_csv2.Time; flipud(data_csv2.Time)],...
             [data_csv2.(sprintf('%s_mean',states{i})) + data_csv2.(sprintf('%s_std',states{i}));...
              flipud(data_csv2.(sprintf('%s_mean',states{i})) - data_csv2.(sprintf('%s_std',states{i})))],...
             colors.SI_ci, 'EdgeColor','none', 'FaceAlpha',area_alpha);
        
        h(4) = plot(data_csv2.Time, data_csv2.(sprintf('%s_true',states{i})),...
            'Color',colors.SI_true, 'LineWidth',line_width);
        
        h(5) = plot(data_csv2.Time, data_csv2.(sprintf('%s_mean',states{i})), '-.',...
            'Color',colors.SI_mean, 'LineWidth',line_width);
        
        if i == 2  % Observations for I
            h(6) = plot(data_csv2.Time, data_csv2.I_observation, '*',...
                'Color',colors.SI_obs, 'MarkerSize',marker_size,...
                'LineWidth',marker_line_width, 'MarkerFaceColor','none');
        end
        
        % ================== Threshold line and annotation ==================
        if i == 1
            yline(S_star, '--', 'Color',colors.threshold,...
                  'LineWidth',threshold_line_width, 'HandleVisibility','off');
            text(t_sim*0.075, S_star, '$S^\star$', 'VerticalAlignment','bottom',...
                 'HorizontalAlignment','right', 'FontSize',font_size,...
                 'Color',colors.threshold, 'Interpreter', 'latex');
        elseif i == 2
            yline(I_max, '--', 'Color',colors.threshold,...
                  'LineWidth',threshold_line_width, 'HandleVisibility','off');
            text(t_sim*0.125, I_max, '$I_{max}$', 'VerticalAlignment','bottom',...
                 'HorizontalAlignment','right', 'FontSize',font_size,...
                 'Color',colors.threshold, 'Interpreter', 'latex');
        end
        
        % ================== Dynamic legend construction ==================
        if i == 2
            valid_handles = h([1,2,3,4,5,6]);
            legend_labels = {'LS-PINNs True','LS-PINNs Est.','LS-PINNs Obs.',...
                            'SI-PINNs True','SI-PINNs Est.','SI-PINNs Obs.'};
            NumColumns = 2;
        else
            valid_handles = h([1,2,4,5]);
            legend_labels = {'LS-PINNs True','LS-PINNs Est.',...
                            'SI-PINNs True','SI-PINNs Est.'};
            NumColumns = 1;
        end
        
    else  % Fourth plot (control u)
        % csv1 control
        h(1) = stairs(data_csv1.Time, data_csv1.u_actual,...
            'Color',colors.LS_mean, 'LineWidth',line_width);
        
        % csv2 control
        h(2) = stairs(data_csv2.Time, data_csv2.u_actual,...
            'Color',colors.SI_mean, 'LineWidth',line_width);
        
        % Max control threshold and annotation
        yline(u_max, '--', 'Color',colors.threshold,...
              'LineWidth',threshold_line_width, 'HandleVisibility','off');
        text(t_sim*0.165, u_max, '$u_{max}$', 'VerticalAlignment','bottom',...
             'HorizontalAlignment','right', 'FontSize',threshold_font_size,...
             'Color',colors.threshold, 'Interpreter', 'latex');
        
        valid_handles = h([1,2]);
        legend_labels = {'LS-PINNs','SI-PINNs'};
        NumColumns = 1;
    end
    
    % ================== Axis settings ==================
    xlim([0 t_sim]);
    if i == 2
        ylim([0 0.15]);
    elseif i == 4
        ylim([0 0.55]);
    else
        ylim([0 1]);
    end
    
    % ================== Legend and formatting ==================
    lgd = legend(valid_handles, legend_labels,...
                'Location', 'best',...
                'FontSize', legend_size,...
                'AutoUpdate', 'off',...
                'Interpreter', 'latex',...
                'NumColumns', NumColumns);
    
    set(gca, 'FontSize',font_size, 'LineWidth',1.5, 'GridAlpha',0.25);

    if i == 3 || i == 4 
    xlabel('Time (days)', 'FontSize',font_size);
    end

    ylabel(ylabels{i}, 'FontSize',font_size, 'Interpreter', 'latex');
    grid on;
    box on;
    hold off;
end

% Adjust subplot spacing
for i = 1:4
    pos = get(subplot_handles(i), 'Position'); % Get current position [left bottom width height]
    % Reduce spacing: increase width/height, shift position
    pos(1) = pos(1) - 0.03; % Left
    pos(2) = pos(2) + 0.01; % Bottom
    pos(3) = pos(3) + 0.03; % Width
    pos(4) = pos(4) + 0.03; % Height
    set(subplot_handles(i), 'Position', pos); % Update
end

% Adjust overall figure size (optional)
set(gcf, 'Position', [0.1, 0.1, 0.8, 0.8]);

%% β 
% Create new figure
figure('Units','normalized','Position',[0.1,0.1,0.8,0.8])

% ========== Subplot 1: Beta parameter comparison ==========
subplot(2,1,1)
hold on

% True curve
p1 = plot(data_csv1.Time, data_csv1.beta_true, '-',...
        'Color',colors.true_beta, 'LineWidth',line_width);

% Estimated curves
p2 = plot(data_csv1.Time, data_csv1.beta_mean, '--',...
        'Color',colors.LS_mean, 'LineWidth',line_width);
p3 = plot(data_csv2.Time, data_csv2.beta_mean, '--',...
        'Color',colors.SI_mean, 'LineWidth',line_width);

% Confidence intervals
fill([data_csv1.Time; flipud(data_csv1.Time)],...
     [data_csv1.beta_mean+data_csv1.beta_std; flipud(data_csv1.beta_mean-data_csv1.beta_std)],...
     colors.LS_ci, 'EdgeColor','none', 'FaceAlpha',area_alpha2)
fill([data_csv2.Time; flipud(data_csv2.Time)],...
     [data_csv2.beta_mean+data_csv2.beta_std; flipud(data_csv2.beta_mean-data_csv2.beta_std)],...
     colors.SI_ci, 'EdgeColor','none', 'FaceAlpha',area_alpha2)

% Formatting
xlim([0 t_sim])
set(gca, 'FontSize',font_size, 'LineWidth',1.5, 'GridAlpha',0.25)
ylabel('$\beta$', 'FontSize',font_size, 'Interpreter', 'latex')
legend([p1 p2 p3], {'True $\beta$ ','LS-PINNs Est.','SI-PINNs Est.'},...
       'Location','southwest', 'FontSize',legend_size, 'Interpreter', 'latex')
grid on
box on
hold off

% ========== Subplot 2: Relative error of Beta ==========
subplot(2,1,2)
hold on

% Compute relative error
csv1_beta_error = abs(data_csv1.beta_mean - data_csv1.beta_true) ./ data_csv1.beta_true;
csv2_beta_error = abs(data_csv2.beta_mean - data_csv2.beta_true) ./ data_csv2.beta_true;

% Error curves
p1 = plot(data_csv1.Time, csv1_beta_error, '-',...
        'Color',colors.LS_mean, 'LineWidth',line_width);
p2 = plot(data_csv2.Time, csv2_beta_error, '-',...
        'Color',colors.SI_mean, 'LineWidth',line_width);

% Formatting
xlim([0 t_sim])
set(gca, 'YScale','log', 'FontSize',font_size, 'LineWidth',1.5)
xlabel('Time (days)', 'FontSize',font_size)
ylabel('RAE of $\hat \beta$', 'FontSize',font_size, 'Interpreter', 'latex')
legend([p1 p2], {'LS-PINNs','SI-PINNs'},...
       'Location','northwest', 'FontSize',legend_size)
grid on
box on
hold off

%% Filtered rMSE for SIR and β
time = data_csv1.Time;

% Set filtering time range
t1 = 0;  % Start time
t2 = 50;  % End time

% Define datasets (name, object, label)
datasets = {
    struct('name', 'data_csv1', 'data', data_csv1, 'label', 'LS-PINNs'), 
    struct('name', 'data_csv2', 'data', data_csv2, 'label', 'SI-PINNs')
};

% Preallocate result storage
results = struct();

for d = 1:length(datasets)
    dataset = datasets{d};
    data = dataset.data;
    
    % Filter time
    time_filter = (data.Time >= t1) & (data.Time <= t2);
    
    % Compute errors for each variable
    results(d).label = dataset.label;
    
    S_true = data.S_true(time_filter);
    S_mean = data.S_mean(time_filter);
    results(d).S = compute_errors(S_true, S_mean);
    
    I_true = data.I_true(time_filter);
    I_mean = data.I_mean(time_filter);
    results(d).I = compute_errors(I_true, I_mean);
    
    R_true = data.R_true(time_filter);
    R_mean = data.R_mean(time_filter);
    results(d).R = compute_errors(R_true, R_mean);
    
    beta_true = data.beta_true(time_filter);
    beta_mean = data.beta_mean(time_filter);
    results(d).beta = compute_errors(beta_true, beta_mean);
    
    results(d).I_max = max(data.I_true);  % Global I_max
end

% Error computation function
function err = compute_errors(true_vals, mean_vals)
    err = struct(...
        'rMSE', sum((mean_vals - true_vals).^2) / sum(true_vals.^2));
end

% Print results
fprintf('Error analysis for time range [%d, %d]:\n', t1, t2);
for d = 1:length(results)
    fprintf('\n=== %s ===\n', results(d).label);
    fprintf('S: rMSE=%.3e  ', results(d).S.rMSE);
    fprintf('I: rMSE=%.3e  ', results(d).I.rMSE);
    fprintf('R: rMSE=%.3e  ', results(d).R.rMSE);
    fprintf('β: rMSE=%.3e  ', results(d).beta.rMSE);
    fprintf('I_max = %.4f\n', results(d).I_max);  % Output global I_max
end
