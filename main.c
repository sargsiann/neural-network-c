#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

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



// O - OUTPUT VALUES OF EACH LAYER IN VECTOR TYPE LIKE [I1,I2, -- IN] with N neurons - Y
// W -  WEIGHTS MATRIX WILL BE [w1 , w2,  ... wn (in column) etc ... M times n1,n2,nn (all in columns)] with output to m neurons
// B - The biases for each will be just [b1,b2  .. bn] and will add up to matrix

// SO THE INPUT FOR THE NEXT LAYER WILL BE I = OXW + B THE FORMULA FOR FEED FORWARDING

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
// The main idea in each neuron will get total from its inputs by vec multiplication of inputs and its weights
// [i1,i2 ... ,in] x [w1,w2, ... ,wn] + BIAS 
// The bias is usefull to avoid 0 cases to not acivateing and some logic
// For example to guess house prices Inputs will be the parametres weights how any characteristic affects the price
// AND THE BIAS WILL BE START PRICE OF EACH HOUSE SO IF WE HAVE NO INPUTS WE HAVES BIAS THAT WILL RETURN JUST START PRICE

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

// The idea of n inputs connected to n nodes to get the total of previous is matrix mul like 
// will get the matrix of totals will be to each node n inputs m outputs
// Y = [i1, i2 ... in] x [w(1)1][w(2)1] .. m []
//                       [w(1)2][w(2)2] .. m []
// 						 [w(1)3][w(2)3] .. m []

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

void	print_mtx(t_matrix *mtx) 
{
	printf("----------------------------\n");
	for (size_t i = 0; i < mtx->rows; i++) {
		for (size_t j = 0; j < mtx->cols; j++) {
			printf("%f ", mtx->mat[i][j]);
		}
		printf("\n");
	}
	printf("----------------------------\n");
}

void	free_mtx(t_matrix *mtx) {
	if (!mtx)
		return;
	for (size_t i = 0; i < mtx->rows; i++) {
		free(mtx->mat[i]);
	}
	free(mtx->mat);
}


// Function for adding padding to matrix
float	**padding_added_mtx(t_matrix *image, int padding, bool flag) 
{
	// Rows and cols are equal
	size_t rows = image->rows + 2 * padding;
	size_t cols = rows;

	printf("Rows: %zu, Cols: %zu\n", rows, cols);
	float	**mtx = (float **)malloc(sizeof(float *) * rows);
	for (size_t i = 0; i < rows;++i) {
		mtx[i] = (float *)malloc(sizeof(float ) * rows);
		for (size_t j = 0; j < cols; ++j) {
			if (j < padding || cols - j <= padding || i < padding || rows - i <= padding)
			{
				mtx[i][j] = 0;
			}
			else 
				mtx[i][j] = image->mat[i - padding][j - padding];
		}
	}
	if (flag)
		free_mtx(image);
	return mtx;
}

float	find_value_kernel(t_matrix *kernel, float **image_mtx, size_t start_row, size_t start_col) 
{
	// 
	float 	**filter = kernel->mat;
	size_t	size = kernel->cols;
	float	res = 0;
	size_t	col = start_col;
	for (size_t i = 0; i < size; i++) {
		// Going back to start col for each row
		start_col = col;
		for (size_t j = 0;j < size;j++) {
			res += image_mtx[start_row][start_col] * filter[i][j];
			printf("%zu%zux%zu%zu + ",start_row, start_col,j, i);
			start_col++;
		}
		// Going to next row
		start_row++;
	}
	printf(" = %f", res);
	printf("\n\n\n");
	return res;
}


// Filtering gets parametres as image mtx , martix of filter, padding size (default zero), and stride size (default one)
t_matrix	filtering(t_matrix *image, t_matrix *kernel, int pding_size, int stride) 
{
	t_matrix res;
	res.mat = NULL;
	
	if (image->cols != image->rows || kernel->cols != kernel->rows)
		return res;
		
	// The matrix rows and cols calculation
	size_t	rows = (image->rows + 2 * pding_size - kernel->rows)/stride + 1;
	size_t	cols = rows;

	res.cols = cols;
	res.rows = rows;
	printf("Rows: %zu, Cols: %zu\n", rows, cols);
	// If wee add padding to our edges wee need to reallocate and free image
	if (pding_size != 0)
		image->mat = padding_added_mtx(image, pding_size, true);

	res.mat = malloc(sizeof(float *) * rows);
	for (size_t i = 0; i < rows; i++) {
		res.mat[i] = malloc(sizeof(float) * cols);
	}
	print_mtx(image);
	size_t	s_row = 0;
	size_t	s_col = 0;
	for (size_t i = 0; i < rows; i++)
	{
		s_row = i * stride;
		for (size_t j = 0; j < cols; j++)
		{
			s_col = j * stride;
			if (j + kernel->cols > image->cols) 
				break;
			res.mat[i][j] = find_value_kernel(kernel, image->mat, s_row, s_col);
			
		}
		if (i + kernel->rows > image->rows)
			break;
	}
	printf("\n\n\n");
	printf("Filtered Matrix \n");
	print_mtx(&res);
}

int main() 
{
	t_matrix mtx;

	mtx.rows = 6;
	mtx.cols = 6;
	mtx.mat = malloc(sizeof(float *) * mtx.rows);
	for (size_t i = 0; i < mtx.rows; i++) {
		mtx.mat[i] = malloc(sizeof(float) * mtx.cols);
		for (size_t j = 0; j < mtx.cols; j++) {
			if (j >= 3) 
				mtx.mat[i][j] = 1;
			else
				mtx.mat[i][j] = 0;
		}
	}
	t_matrix kernel;

	kernel.rows = 3;
	kernel.cols = 3;

	kernel.mat = malloc(sizeof(float *) * kernel.rows);
	for (size_t i = 0; i < kernel.rows; i++) {
		kernel.mat[i] = malloc(sizeof(float) * kernel.cols);
		for (size_t j = 0; j < kernel.cols; j++) {
			if (j == 0)
				kernel.mat[i][j] = 1.0;
			else if (j == 2)
				kernel.mat[i][j] = -1.0;
			else
				kernel.mat[i][j] = 0.0;
		}
	}
	printf("\n\n\nMatrix of filter \n");
	print_mtx(&kernel);

	filtering(&mtx,&kernel, 0, 1);
	return 0;
}