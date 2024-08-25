# # # # # # # # # # # # # # # # # # # # # #
# # Building fftw/3.3.10.5
# # Not required on Perlmutter as we can
# # use cray-fftw/3.3.10.5 for heFFTe
# # # # # # # # # # # # # # # # # # # # # #

First build FFTW 3.10 with double and single precision

wget https://www.fftw.org/fftw-3.3.10.tar.gz

tar xvzf fftw-3.3.10.tar.gz

mv fftw-3.3.10 fftw-3.3.10-src

mkdir fftw-3.3.10

./configure --prefix=/global/homes/a/atif/packages/fftw-3.3.10 --enable-mpi --enable-openmp --enable-shared CC=gcc CXX=g++ CFLAGS=-fPIC CXXFLAGS=-fPIC --enable-float

make -j16

make install

(base) atif@perlmutter:login17:~/packages/fftw-3.3.10-src> ls ../fftw-3.3.10/bin/
fftwf-wisdom  fftw-wisdom-to-conf
(base) atif@perlmutter:login17:~/packages/fftw-3.3.10-src> ls ../fftw-3.3.10/include/
fftw3.f  fftw3.f03  fftw3.h  fftw3l.f03  fftw3l-mpi.f03  fftw3-mpi.f03	fftw3-mpi.h  fftw3q.f03
(base) atif@perlmutter:login17:~/packages/fftw-3.3.10-src> ls ../fftw-3.3.10/lib/
cmake	      libfftw3f_mpi.a	libfftw3f_mpi.so.3	 libfftw3f_omp.la    libfftw3f_omp.so.3.6.10  libfftw3f.so.3.6.10
libfftw3f.a   libfftw3f_mpi.la	libfftw3f_mpi.so.3.6.10  libfftw3f_omp.so    libfftw3f.so	      pkgconfig
libfftw3f.la  libfftw3f_mpi.so	libfftw3f_omp.a		 libfftw3f_omp.so.3  libfftw3f.so.3

make clean

./configure --prefix=/global/homes/a/atif/packages/fftw-3.3.10 --enable-mpi --enable-openmp --enable-shared CC=gcc CXX=g++ CFLAGS=-fPIC CXXFLAGS=-fPIC

make -j16

make install

This is what the install directory should look like

(base) atif@perlmutter:login17:~/packages/fftw-3.3.10-src> ls ../fftw-3.3.10/bin/
fftwf-wisdom  fftw-wisdom  fftw-wisdom-to-conf
(base) atif@perlmutter:login17:~/packages/fftw-3.3.10-src> ls ../fftw-3.3.10/include/
fftw3.f  fftw3.f03  fftw3.h  fftw3l.f03  fftw3l-mpi.f03  fftw3-mpi.f03	fftw3-mpi.h  fftw3q.f03
(base) atif@perlmutter:login17:~/packages/fftw-3.3.10-src> ls ../fftw-3.3.10/lib/
cmake		 libfftw3f_mpi.la	  libfftw3f_omp.la	   libfftw3f.so.3	libfftw3_mpi.so		libfftw3_omp.so		libfftw3.so.3.6.10
libfftw3.a	 libfftw3f_mpi.so	  libfftw3f_omp.so	   libfftw3f.so.3.6.10	libfftw3_mpi.so.3	libfftw3_omp.so.3	pkgconfig
libfftw3f.a	 libfftw3f_mpi.so.3	  libfftw3f_omp.so.3	   libfftw3.la		libfftw3_mpi.so.3.6.10	libfftw3_omp.so.3.6.10
libfftw3f.la	 libfftw3f_mpi.so.3.6.10  libfftw3f_omp.so.3.6.10  libfftw3_mpi.a	libfftw3_omp.a		libfftw3.so
libfftw3f_mpi.a  libfftw3f_omp.a	  libfftw3f.so		   libfftw3_mpi.la	libfftw3_omp.la		libfftw3.so.3


# # # # # # # # # # # # # # # # # # # # # #
# # Building HeFFTe with cray-fftw/3.3.10.5
# # # # # # # # # # # # # # # # # # # # # #

git clone https://github.com/icl-utk-edu/heffte.git

mv heffte heffte-src

mkdir heffte-2.4.0

cd heffte-src

mkdir build

cd build

module load cray-fftw/3.3.10.5

cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/global/homes/a/atif/packages/heffte-2.4.0 -DHeffte_ENABLE_FFTW=ON -DHeffte_ENABLE_CUDA=OFF -DFFTW_ROOT=/opt/cray/pe/fftw/3.3.10.5/x86_milan/ -DFFTW_INCLUDES=/opt/cray/pe/fftw/3.3.10.5/x86_milan/include/ -DCMAKE_CXX_FLAGS="-L/opt/cray/pe/fftw/3.3.10.5/x86_milan/lib -fopenmp -lfftw3 -lfftw3f -lfftw3f_mpi -lfftw3f_omp -lfftw3_mpi -lfftw3_omp -I/opt/cray/pe/fftw/3.3.10.5/x86_milan/include" ..

make -j16

make install


