# MatMul
 В данной лабораторной работе было распараллелено перемножение матриц, каждая нить отвечает за один элемент результирующей  матирцы и высчитывает его как скалярное произведение.

# Характеристики
 CPU: Intel Core i7 2600 3.40 GHz
 GPU: Geforce GTX 1060 6gb
 RAM: 16 Gb
 CUDA: 10.1

# Результаты

 В результатах размеры матриц N=n=m=k
 |N|время на GPU, мс|время на CPU, мс|Ускорение|
 |:---:|:----:|:----:|:-:|
 |100|0.7223392|2.4|3.322539|
 |250|3,8693696|45.9|11,86239743|
 |500|25,865203|432.8|11,86239743|
 |1000|190,5868333|7485,5|39,27606053|
 |2000|1366,088582|63635|46,58189874|
