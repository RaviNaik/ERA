<reponame>A-tilol/leetcode<filename>MaximalSquare/main.go
package main

import "fmt"

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// dp
func maximalSquare(matrix [][]byte) int {
	if len(matrix) == 0 {
		return 0
	}

	side := 0
	for i := range matrix {
		for j := range matrix[0] {
			matrix[i][j] -= '0'
			if matrix[i][j] == 0 || i == 0 || j == 0 {
				side = max(side, int(matrix[i][j]))
				continue
			}

			if matrix[i-1][j] == matrix[i-1][j-1] && matrix[i-1][j-1] == matrix[i][j-1] {
				matrix[i][j] += matrix[i-1][j]
			} else {
				matrix[i][j] += byte(min(int(matrix[i-1][j]), min(int(matrix[i-1][j-1]), int(matrix[i][j-1]))))
			}
			side = max(side, int(matrix[i][j]))
		}
	}

	return side * side
}

// no additional space complexity
func maximalSquare2(matrix [][]byte) int {
	if len(matrix) == 0 {
		return 0
	}

	m, n := len(matrix), len(matrix[0])
	for i := range matrix {
		for j := range matrix[0] {
			matrix[i][j] -= '0'
			if i > 0 {
				matrix[i][j] += matrix[i-1][j]
			}
			if j > 0 {
				matrix[i][j] += matrix[i][j-1]
			}
			if i > 0 && j > 0 {
				matrix[i][j] -= matrix[i-1][j-1]
			}
		}
	}

	side := 0
	for i := range matrix {
		if i+side >= m {
			break
		}
		for j := range matrix[i] {
			if j+side >= n {
				break
			}
			for k := side; i+k < m && j+k < n; k++ {
				rd := matrix[i+k][j+k]
				if i > 0 && j > 0 {
					rd += matrix[i-1][j-1]
				}
				if i > 0 {
					rd -= matrix[i-1][j+k]
				}
				if j > 0 {
					rd -= matrix[i+k][j-1]
				}
				if int(rd) == (k+1)*(k+1) {
					side = k + 1
					// fmt.Println(i, j, k+1, "mij", matrix[i][j], "rd", matrix[i+k][j+k])
				}
			}
		}
	}

	return side * side
}

func maximalSquare1(matrix [][]byte) int {
	if len(matrix) == 0 {
		return 0
	}

	m, n := len(matrix), len(matrix[0])
	rui := make([][]int, m)
	for i := range matrix {
		rui[i] = make([]int, n)
		for j := range matrix[0] {
			if i > 0 {
				rui[i][j] += rui[i-1][j]
			}
			if j > 0 {
				rui[i][j] += rui[i][j-1]
			}
			if i > 0 && j > 0 {
				rui[i][j] -= rui[i-1][j-1]
			}
			if matrix[i][j] == '1' {
				rui[i][j]++
			}
		}
	}

	side := 0
	for i := range rui {
		if i+side >= m {
			break
		}
		for j := range rui[i] {
			if j+side >= n {
				break
			}
			for k := side; i+k < m && j+k < n; k++ {
				rd := rui[i+k][j+k]
				if i > 0 {
					rd -= rui[i-1][j+k]
				}
				if j > 0 {
					rd -= rui[i+k][j-1]
				}
				if i > 0 && j > 0 {
					rd += rui[i-1][j-1]
				}
				if rd == (k+1)*(k+1) {
					side = k + 1
				}
			}
		}
	}

	return side * side
}

func main() {
	fmt.Println(maximalSquare([][]byte{
		[]byte{'1', '0', '1', '0', '0'},
		[]byte{'1', '0', '1', '1', '1'},
		[]byte{'1', '1', '1', '1', '1'},
		[]byte{'1', '0', '0', '1', '0'},
	}))
	fmt.Println(maximalSquare([][]byte{
		[]byte{'0', '1'},
	}))
	fmt.Println(maximalSquare([][]byte{
		[]byte{'0', '1', '0', '1', '0', '1', '0', '1', '1', '0', '1', '1', '0', '1', '0'},
		[]byte{'0', '1', '1', '1', '1', '1', '0', '1', '1', '0', '1', '1', '1', '1', '0'},
		[]byte{'0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '0', '1', '1'},
		[]byte{'1', '0', '1', '1', '0', '1', '0', '0', '1', '1', '1', '0', '0', '0', '1'},
		[]byte{'0', '1', '0', '1', '1', '1', '0', '1', '0', '1', '1', '0', '0', '1', '1'},
		[]byte{'1', '1', '1', '1', '0', '1', '1', '0', '1', '1', '1', '0', '1', '1', '1'},
		[]byte{'1', '1', '1', '1', '0', '1', '0', '1', '0', '1', '1', '1', '1', '1', '0'},
		[]byte{'0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '0'},
		[]byte{'0', '1', '1', '1', '0', '1', '0', '0', '0', '0', '1', '1', '1', '1', '1'},
		[]byte{'1', '1', '1', '1', '1', '0', '1', '0', '1', '1', '1', '0', '0', '1', '1'},
		[]byte{'0', '1', '1', '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1'},
		[]byte{'1', '1', '1', '0', '0', '0', '1', '1', '1', '0', '1', '0', '0', '1', '1'},
		[]byte{'1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '1', '1', '1', '1'},
		[]byte{'1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1'},
		[]byte{'1', '1', '1', '0', '1', '0', '1', '0', '1', '0', '1', '1', '1', '1', '1'},
	}))
}
