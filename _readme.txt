\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
+--------------------------Information------------------------------+
|Date:        2017-06-02                                            |
|University:  Southwest Jiaotong University                         |
|School:      School of Information Science and Technology          |
|Department:  Department of Computing                               |
|Major:       Internet of Things Engineering                        |
|Author:      Lyu, Zhaoyan                                          |
+-------------------------------------------------------------------+
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
+-------------------Instruction to Python files---------------------+
|                                                                   |
|reciprocal_normal_distribution_pdf_cdf.py                          |
|Generate the pdf and cdf of a reciprocal normal distribution       |
|according to formula.                                              |
|-------------------------------------------------------------------|
|                                                                   |
|reciprocal_normal_distribution.py                                  |
|Generate the pdf of a reciprocal normal distribution according     |
|to formula. Spot the reciprocal of mean of normal distribution     |
|and peak value of new distribution.                                |
|-------------------------------------------------------------------|
|                                                                   |
|reciprocal_normal_distribution_test.py                             |
|Generate the pdf of a reciprocal normal distribution by testing.   |
|Generate 100000 random numbers and get their reciprocal. Index     |
|them into 100 boxes and count the number of numbers in each box.   |
|Plot boxes and numbers, get pdf.                                   |
|-------------------------------------------------------------------|
|                                                                   |
|MAP.py                                                             |
|MAP algorithm using convolve to calculate Yi.                      |
|-------------------------------------------------------------------|
|                                                                   |
|MAP_norm.py                                                        |
|MAP algorithm using normal distribution as Xi's distribution.      |
|Using convolve to calculate Yi                                     |
|-------------------------------------------------------------------|
|                                                                   |
|MAP_norm_simplified.py                                             |
|MAP algorithm using normal distribution as Xi's distribution.      |
|Using convolve character of normal distribution to calculate Yi.   |
|-------------------------------------------------------------------|
|                                                                   |
|Extreme_download_probability.py                                    |
|MAP algorithm without 'algorithm 1'. i.e. caching every chunk in   |
|every EN. Shows that the probability model have a predict limit.   |
|-------------------------------------------------------------------|
|                                                                   |
|fit_2.py                                                           |
|Fitting a reciprocal normal distribution into a normal             |
|distribution using a test method. Generate 100000 random numbers   |
|according to reciprocal normal distribution. Then using            |
|scipy.stats.fit() method to fit generated numbers.                 |
|Plot the graph and calculate the error rate.                       |
|-------------------------------------------------------------------|
|                                                                   |
|fit_3.py                                                           |
|Fitting a normal reciprocal distribution into a normal             |
|distribution. The reciprocal of mean of normal distribution is     |
|fit distribution's mean. Calculate variation from mean.            |
|Plot the graph and calculate the error rate.                       |
|-------------------------------------------------------------------|
|                                                                   |
|fit_4.py                                                           |
|Fitting a normal reciprocal distribution into a normal             |
|distribution. The reciprocal of mean of normal distribution is     |
|fit distribution's mean. The normal reciprocal distribution and    |
|fit distribution have same peak value.                             |
|Plot the graph and calculate the error rate.                       |
|-------------------------------------------------------------------|
|                                                                   |
|fit_5                                                              |
|Fitting a normal reciprocal distribution into a normal             |
|distribution. The peak value's spot of reciprocal normal           |
|distribution is fit distribution mean value's spot. The normal     |
|reciprocal distribution and fit distribution have same peak        |
|value. Plot the graph and calculate the error rate.                |
|-------------------------------------------------------------------|
|                                                                   |
|ACBC.py                                                            |
|ACBC algorithm.                                                    |
|-------------------------------------------------------------------|
|                                                                   |
|demo.gui.py                                                        |
|ACBC algorithm demo GUI.                                           |
|-------------------------------------------------------------------|
|                                                                   |
|demo.ACBC                                                          |
|Demo's background algorithm.                                       |
|-------------------------------------------------------------------|
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
