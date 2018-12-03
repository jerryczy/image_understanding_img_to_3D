function read_images(path)
    dirlist = dir(path);
    for i = 1:length(dirlist)
        if length(dirlist(i).name) <= 4
            continue;
        end
        if strcmp(dirlist(i).name(end-3:end), ".jpg")
            img_path = strcat(path, dirlist(i).name);
            disp(img_path)
            img = imread(img_path);
            img = single(rgb2gray(img));
            [f, d] = vl_sift(img);
            if size(f, 2) == 0
                continue;
            end
            data_path = strcat(path ,dirlist(i).name(1:end-3), 'csv');
            dlmwrite(data_path, f);
            dlmwrite(data_path, d, '-append');
        end
    end
end