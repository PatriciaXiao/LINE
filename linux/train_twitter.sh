#!/bin/sh

#g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result line.cpp -o line -lgsl -lm -lgslcblas
#g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result reconstruct.cpp -o reconstruct
#g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result normalize.cpp -o normalize
#g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result concatenate.cpp -o concatenate

python3 preprocess_twitter.py twitter-links.txt net_twitter.txt
./reconstruct -train net_twitter.txt -output net_twitter_dense.txt -depth 2 -threshold 1000
./line -train net_twitter_dense.txt -output twitter_vec_1st_wo_norm.txt -binary 1 -size 128 -order 1 -negative 5 -samples 10000 -threads 40
./line -train net_twitter_dense.txt -output twitter_vec_2nd_wo_norm.txt -binary 1 -size 128 -order 2 -negative 5 -samples 10000 -threads 40
./normalize -input twitter_vec_1st_wo_norm.txt -output twitter_vec_1st.txt -binary 1
./normalize -input twitter_vec_2nd_wo_norm.txt -output twitter_vec_2nd.txt -binary 1
./concatenate -input1 twitter_vec_1st.txt -input2 twitter_vec_2nd.txt -output twitter_vec_all.txt -binary 1

cd evaluate
./run.sh ../twitter_vec_all.txt
python3 score.py twitter_result.txt
cd ..
