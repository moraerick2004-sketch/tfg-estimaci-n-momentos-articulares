%% convert_to_csv.m
% Primer paso del pipeline: convierte las tablas .mat de Camargo et al. (2021)
% a CSV para poder leerlas desde Python. Ejecutar en MATLAB.

% === EDITA ESTAS RUTAS ===
dataset_dir = '';   % carpeta con los sujetos AB* del dataset de Camargo
output_dir  = '';   % carpeta donde se escriben los CSV

% Sensores que necesito: IMU (entrada), cinematica inversa y dinamica inversa.
sensor_types = {'imu', 'ik', 'id'};
modes = {'treadmill', 'levelground', 'stair', 'ramp'};

subjects = dir(fullfile(dataset_dir, 'AB*'));

for s = 1:length(subjects)
    subject = subjects(s).name;
    subject_path = fullfile(dataset_dir, subject);

    % Cada sujeto tiene una carpeta de fecha; cojo la primera (ignoro osimxml).
    date_dirs = dir(subject_path);
    date_dirs = date_dirs([date_dirs.isdir]);
    date_dirs = date_dirs(~ismember({date_dirs.name}, {'.', '..', 'osimxml'}));

    if isempty(date_dirs)
        fprintf('WARNING: No date folder found for %s\n', subject);
        continue;
    end

    date_folder = date_dirs(1).name;
    session_path = fullfile(subject_path, date_folder);

    fprintf('Processing %s/%s...\n', subject, date_folder);

    for m = 1:length(modes)
        mode = modes{m};
        mode_path = fullfile(session_path, mode);

        if ~isfolder(mode_path)
            continue;
        end

        % La lista de ensayos la saco de la carpeta imu de cada modo.
        imu_path = fullfile(mode_path, 'imu');
        if ~isfolder(imu_path)
            continue;
        end

        mat_files = dir(fullfile(imu_path, '*.mat'));

        for t = 1:length(mat_files)
            trial_name = mat_files(t).name;
            trial_base = trial_name(1:end-4);   % quita la extension .mat

            out_trial_dir = fullfile(output_dir, subject, [mode '_' trial_base]);
            if ~isfolder(out_trial_dir)
                mkdir(out_trial_dir);
            end

            for st = 1:length(sensor_types)
                sensor = sensor_types{st};
                mat_path = fullfile(mode_path, sensor, trial_name);

                if ~isfile(mat_path)
                    fprintf('  SKIP: %s not found\n', mat_path);
                    continue;
                end

                try
                    loaded = load(mat_path);

                    % El .mat guarda una sola tabla; cojo el primer campo.
                    fields = fieldnames(loaded);
                    var_name = fields{1};
                    tbl = loaded.(var_name);

                    csv_path = fullfile(out_trial_dir, [sensor '.csv']);
                    writetable(tbl, csv_path);

                catch ME
                    fprintf('  ERROR: %s - %s\n', mat_path, ME.message);
                end
            end

            fprintf('  Converted: %s/%s/%s\n', subject, mode, trial_base);
        end
    end
end

fprintf('\nDone! CSV files saved to: %s\n', output_dir);
