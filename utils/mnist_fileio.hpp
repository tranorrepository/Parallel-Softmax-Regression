#ifndef MNIST_FILEIO
#define MNIST_FILEIO

#include <fstream>
#include <iostream>
//#include <cstdio>
#include <stdlib.h>

using namespace std;
int reverseInt (int i);


//read a text file and insert lines in vector
//return number of lines read; 0 if file failed to open
template <typename Type>
size_t readData(size_t size, const char* file_name, Type* vector) {

    std::ifstream source_file;
    source_file.open(file_name);

    if (!source_file.is_open()) {
        cerr << "Failed to open file " << file_name << endl;
        return 0;
    }

    size_t i = 0;
    for (Type val = 0; i < size && source_file >> val; i++) {
        vector[i] = val;
    }

    source_file.close();
    cout << "Finished reading " << i << " lines of " << file_name << "\n";
    return i;
}

//write content of a vector to a text file, separated by \n
template <typename Type>
void writeData(size_t size, Type* vector, const char* file_name = "output.txt") {

    std::ofstream output_file;
    output_file.open(file_name, ios_base::trunc);

    if (!output_file.is_open()) {
        cerr << "Failed to open file " << file_name << endl;
        return;
    }

    size_t i;
    for (i = 0; i < size; i++) {
        output_file << vector[i] << "\n";
    }

    output_file.close();
    cout << "Finished writing " << i << " lines of " << file_name << "\n";
}

template <typename Type>
size_t readImageData(size_t size, const char* file_name, Type* vector)
{
    ifstream data(file_name,ios::in | ios::binary);

    if (data.is_open())
    {
        //mnist data structure magic number->sample number->rows->columns->pixel value
        int magic_number,number_images,n_rows,n_cols;
        // Start reading Database
        // First read magic number, number of images, rows and
        // columns of the pixel 
        data.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        data.read((char*)&number_images,sizeof(number_images));
        number_images= reverseInt(number_images);
        data.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        data.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        cout<<"n_rows number: "<<n_rows<<endl;
        cout<<"n_cols number: "<<n_rows<<endl;
        //reading pixel value
        size_t i;
        for(i=0; i<size; i++){
            
            unsigned char temp=0;   
            data.read((char*)&temp,sizeof(temp));
            //feature scaling normalized by 255
            vector[i]=(Type)temp/255;
        
        }

        data.close();
        cout << "Finished reading " << i << " data from " << file_name << "\n";
        return i;
    }
    else{ 
        cout<<"error reading image_database\n";    
        return 0;
    }
}

template <typename Type>
size_t readImageLables(size_t size, const char* file_name, Type* vector)
{
    //mnist data structure magic number->sample number->pixel value
    ifstream lables(file_name,ios::in | ios::binary);

    if (lables.is_open())
    {
        int magic_number,number_images;
        unsigned char lable;

        // Start reading Database
        lables.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);

        lables.read((char*)&number_images,sizeof(number_images));
        number_images= reverseInt(number_images);

        //debug use
        // cout<<"magic number: "<<magic_number<<"\n";
        // cout<<"image number"<<number_images<<"\n";
        
        size_t i;
        for(i=0;i<size;i++)
        {
            lables.read((char*)&lable,sizeof(lable));
            vector[i]=(Type)lable;
			//cout<<vector[i]<<endl;
        }
        lables.close();
        cout << "Finished reading " << i << " label from " << file_name << "\n";
        return i;
    }
    else {
        cout<<"error reading lables_database\n";
        return 0;
    }
}



int reverseInt (int i) //reverse a int from high-endian to low-endian
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

#endif
