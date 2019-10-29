#pragma once


template <typename T>
class vector
{
private:

	bool host = false;
public:
	int length;
	T* arr;
	vector* cudavector;
	T* cudaarr;
	vector(int length):length(length)
	{
		arr = new T[length];
	}
	vector(int length,T* data):length(length),arr(data)
	{

	}


};

