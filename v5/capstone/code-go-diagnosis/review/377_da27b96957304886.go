<reponame>yukibear/algorithms<filename>leetcode/solutions/0119.PascalsTriangleII/getRow.go
package solution

func getRow(rowIndex int) []int {
	lastRow := []int{1, 1}

	for i := 0; i <= rowIndex; i++ {
		newRow := make([]int, i+1)
		newRow[0] = 1

		for j := 1; j < i; j++ {
			newRow[j] = lastRow[j-1] + lastRow[j]
		}

		newRow[len(newRow)-1] = 1
		lastRow = newRow
	}

	return lastRow
}
