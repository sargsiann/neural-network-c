#include <stdio.h>
#include <math.h>

# define MATH_E  2.71828

// Struct of vector
typedef struct s_vec {
	float *vec;
	size_t size;
}	t_vec;

// Struct of matrix
typedef struct s_matrix  {
	float **mat;
	size_t rows;
	size_t cols;
}	t_matrix;



// ACTIVATION FUNCTIONS

// if the NN was linear and with no activ functions it could solve only cases like of all lights or some lights are not lighting
// if we have activ functions we can make it more comperhensive to guess cases when some lights light more higher
// SOME NOT SOME MORE LOWER SO MAKING MORE VARIETY CASES

// The relu acitvation functions getting from max(0,total) 
// Values 0 -> +inf if total is minus we can get dying relu which makes more bad for that cases
// Useful because of more less cpu cost , and is useful in cases of higher ranges value to detect from lowest to highes better 
// than sigmoid or tanh
float	ReLu(float total) 
{
	if (total < 0) 
		return 0;
	return total;
}


// The sigmoid Activation function converts total sum of weights input of neuron to formula
// 1/(1 + e^total) which makes it from 0 to 1 to output 
// Has issues of saturation so in very little or high cases the output is veryy little differ
float	Sigmoid(float total) {
	return (1/(1 + pow(MATH_E,total)));
}

// The medium range values of inputs gots to 0 
// -1 the less 1 the most this makes more range to get the edge cases
// output values from -1 to 1 which makes NN to learn more good
float	TanH(float total) {
	return (2 * Sigmoid(2 * total) - 1);
}


// SoftMax getts the vector of input values and return the probabilty from 0 -> 1 so what is the most likely
// Ussually using in some gueessers like what in picture
// The getting of probabilty is by the formula e^node[i] / Sum(e^node[n]) 0->n
t_vec	SoftMax(t_vec *vector) 
{
	// Creating an array
	t_vec probs;

	probs.vec = malloc(sizeof(float) * vector->size);

	// Getting the sum for dividing
	float sum = 0;
	for (int i = 0; i < vector->size; i++) {
		sum += pow(MATH_E,vector->vec[i]);
	}

	// And filling to new vector of probs
	for (int i = 0; i < vector->size; i++) {
		probs.vec[i] = pow(MATH_E,vector->vec[i]) / sum;
	}

	return probs;
}

// the more beatiful relu in case 0 the func is differentive which makes more advanced that relu and more smooth
float	SoftPlus(float total) {
	return (log10f(1 + pow(MATH_E,total)));
}


// Multiplications of two vectors (random sized)
t_vec vectors_multiply(t_vec v, t_vec m) {
	if (v.size != m.size) 
		return (t_vec){NULL, 0};
	t_vec result;	

	result.vec = malloc(sizeof(float) * v.size);
	result.size = v.size;
	if (!result.vec)
		return (t_vec){NULL, 0};
	for (size_t i = 0; i < v.size; i++) {
		result.vec[i] = v.vec[i] * m.vec[i];
	}
	return result;
}

t_matrix matrix_multiply(t_matrix a, t_matrix b) {
	if (a.cols != b.rows) 
		return (t_matrix){NULL, 0, 0};
	t_matrix result;
	result.rows = a.rows;
	result.cols = b.cols;
	result.mat = malloc(sizeof(float *) * result.rows);
	if (!result.mat)
		return (t_matrix){NULL, 0, 0};
	for (size_t i = 0; i < result.rows; i++) {
		result.mat[i] = malloc(sizeof(float) * result.cols);
		if (!result.mat[i]) {
			for (size_t j = 0; j < i; j++)
				free(result.mat[j]);
			free(result.mat);
			return (t_matrix){NULL, 0, 0};
		}
		for (size_t j = 0; j < result.cols; j++) {
			result.mat[i][j] = 0;
			for (size_t k = 0; k < a.cols; k++) {
				result.mat[i][j] += a.mat[i][k] * b.mat[k][j];
			}
		}
	}
	return result;
}

// In process
t_matrix	*transpose_mtx(t_matrix *a) {
	t_matrix *new = malloc(sizeof(t_matrix *) * a->cols);
	if (!new)
		return NULL;
	new->rows = a->cols;
	new->cols = a->rows;
	new->mat = malloc(sizeof(float *) * new->rows);
	for (size_t i = 0; i < new->rows; i++) {
		new->mat[i] = malloc(sizeof(float) * new->cols);
		for (size_t j = 0;j < new->cols; j++) {
			new->mat[i][j] = a->mat[j][i];
		}
	}
	return new;
}