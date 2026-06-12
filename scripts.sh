python main.py \
    --codec jpeg \
    --input "fs_0045_3T.h5" \
    --output results/jpeg_q10 \
    --quality 20


python main.py \
    --codec jpeg2000 \
    --input "fs_0045_3T.h5" \
    --output results/jp2_r100 \
    --cratio 150

python main.py \
    --codec h264 \
    --input "fs_0045_3T.h5" \
    --output results/h264_crf45 \
    --crf 45


python main.py \
    --codec h265 \
    --input "fs_0045_3T.h5" \
    --output results/h265_crf45 \
    --crf 45

python eval_rd.py \
    --files "fs_0045_3T.h5"


python -u run_rd_curves_V2.py --files fs_0045_3T.h5 fs_0095_1_5T.h5 fs_0069_1_5T.h5
