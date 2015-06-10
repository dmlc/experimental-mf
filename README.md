# Fast Matrix Factorization

## Features
   * Cache-friendly Multithread Matrix Factorization.
   * Fast Multithread Stochastic Gradient Langevin Dynamics (SGLD) for Matrix Factorization.
   * Fast Multithread Differentially Private Matrix Factorization.
   * Matrix Factorization with Adaptive Regularizer.

## Data
    Google's Protobuf as input. Use data/getdata.cc to convert from userwise raw data to protobuf:
    ```bash
    ./getdata -r [userwise_raw_data] -w [protobuf_binary] --method [protobuf] --size [int]
    ```
    A sample of userwise raw data looks like:
    ```
    0:
    11,5.0
    21,3.0
    1:
    9,5.0
    12,1.0
    ```
    where there are two users '0' and '1'.

    Or if you only have rating wise raw data, you can first convert to a userwise raw data:
    ```bash
    ./getdata -r [rating_wise_raw] -w [userwise_raw_data] --method [userwise] --split [int]
    ```

    A sample of rating_wise_raw data with a header looks like:
    ```
    100000
    0,1,5.0
    0,2,1.0
    ```
    where the header indicates the number of ratings, and follows by user_id, item_id and rating in each line.

## Environment Requirment
    * GCC 4.9 or higher
    ```bash
    tar zxf gcc-4.9.2.tar.gz;cd gcc-4.9.2;contrib/download_prerequisites;cd ..;mkdir buildc;cd buildc;../gcc-4.9.2/configure --disable-multilib;make -j 32;sudo make install;cd ..;
    ```
    * Intel TBB
    ```bash
    sudo apt-get install libtbb-dev
    ```
    * Google Protobuf
    ```bash
    sudo apt-get install -y libprotobuf-dev; sudo apt-get install -y protobuf-compiler;
    ```
    * Intel MKL


## Reference
[1] [Fast Differentially Private Matrix Factorization](http://arxiv.org/abs/1505.01419). Ziqi Liu, Yu-Xiang Wang, Alex Smola.
[2] Learning recommender systems with adaptive regularization. Steffen Rendle.