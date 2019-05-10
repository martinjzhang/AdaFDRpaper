python ./debug/debug_benchmark.py -d 'load_airway' -o 'airway' &
python ./debug/debug_benchmark.py -d 'load_bottomly' -o 'bottomly' &
python ./debug/debug_benchmark.py -d 'fmri_auditory' -o 'fmri_auditory' &
python ./debug/debug_benchmark.py -d 'fmri_imagination' -o 'fmri_imagination' &
# python ./debug/debug_benchmark.py -d 'load_pasilla' -o 'pasilla' &
# python ./debug/debug_benchmark.py -d 'load_2DGM' -o '2DGM_NeuralFDR' &
python ./debug/debug_benchmark.py -d 'load_5DGM' -o '5DGM_NeuralFDR' &
# python ./debug/debug_benchmark.py -d 'load_GTEx_full' -o 'GTEx_NeuralFDR' &