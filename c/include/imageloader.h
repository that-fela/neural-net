#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <png.h>
#include <string.h>

typedef struct {
    float *images;
    float *labels;
} ImageData;

ImageData get_images() {
    float *images = (float*)malloc(10 * 400 * sizeof(float));
    float *labels = (float*)malloc(10 * 4 * sizeof(float));

    struct dirent *entry;
    DIR *dp = opendir("/home/leo/dev/neural-net/test-data/my-hand-written-numbers"); // "." represents the current directory

    if (dp == NULL) {
        perror("opendir");
        exit(1);
    }

    int width, height;
    png_byte color_type;
    png_byte bit_depth;
    png_bytep *row_pointers = NULL;

    int loop_index = 0;
    while ((entry = readdir(dp))) {
        if (entry->d_type == DT_REG) { // Check if it's a regular file
            printf("%s\n", entry->d_name);

            char *path = "/home/leo/dev/neural-net/test-data/my-hand-written-numbers";
            char *filename = entry->d_name;
            char *full_path = malloc(strlen(path) + strlen(filename) + 2);
            memccpy(full_path, path, strlen(path) + 1, strlen(path));
            memccpy(full_path + strlen(path), "/", 1, 1);
            memccpy(full_path + strlen(path) + 1, filename, strlen(filename) + 1, strlen(filename));

            printf("%s\n", full_path);

            FILE *fp = fopen(full_path, "rb");

            png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
            if(!png) abort();

            png_infop info = png_create_info_struct(png);
            if(!info) abort();

            if(setjmp(png_jmpbuf(png))) abort();

            png_init_io(png, fp);

            png_read_info(png, info);

            width      = png_get_image_width(png, info);
            height     = png_get_image_height(png, info);
            color_type = png_get_color_type(png, info);
            bit_depth  = png_get_bit_depth(png, info);

            // Read any color_type into 8bit depth, RGBA format.
            // See http://www.libpng.org/pub/png/libpng-manual.txt

            if(bit_depth == 16)
                png_set_strip_16(png);

            if(color_type == PNG_COLOR_TYPE_PALETTE)
                png_set_palette_to_rgb(png);

            // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
            if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
                png_set_expand_gray_1_2_4_to_8(png);

            if(png_get_valid(png, info, PNG_INFO_tRNS))
                png_set_tRNS_to_alpha(png);

            // These color_type don't have an alpha channel then fill it with 0xff.
            if(color_type == PNG_COLOR_TYPE_RGB ||
                color_type == PNG_COLOR_TYPE_GRAY ||
                color_type == PNG_COLOR_TYPE_PALETTE)
                png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

            if(color_type == PNG_COLOR_TYPE_GRAY ||
                color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
                png_set_gray_to_rgb(png);

            png_read_update_info(png, info);

            row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
            for(int y = 0; y < height; y++) {
                row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
            }

            png_read_image(png, row_pointers);

            // fclose(fp);

            // float *float_array = (float *)malloc(20 * 20 * 3 * sizeof(float));

            // for(int i = 0; i < 20; i++) {
            //     for(int j = 0; j < 20; j++) {
            //         for(int k = 0; k < 3; k++) {
            //             float_array[i * 20 * 3 + j * 3 + k] = row_data[i * png_get_rowbytes(png, info) + j * 3 + k] / 255.0;
            //         }   
            //     }
            // }

            // for(int i = 0; i < 20; i++) {
            //     for(int j = 0; j < 20; j++) {
            //         images[loop_index * 400 + i * 20 + j] = float_array[i * 20 * 3 + j * 3];
            //     }
            // }


            // read the file name and get the label.
            // and make binary array for the label
            // eg 2.png => 2 => "0010"
            char letter = filename[0];
            int label = letter - '0';
            for(int i = 0; i < 4; i++) {
                if(i == label) {
                    labels[loop_index * 4 + i] = 1.0;
                } else {
                    labels[loop_index * 4 + i] = 0.0;
                }
            }

            loop_index++;
        }   
    }

    closedir(dp);
}
