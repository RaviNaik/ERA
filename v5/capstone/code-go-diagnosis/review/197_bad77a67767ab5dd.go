package Sorting

func BubbleSort(arr []int) {
	n := len(arr)
	flag := true
	for i := 0; flag && i < n; i++ {
		flag = false
		for j := 0; j < n-i-1; j++ {
			if arr[j] > arr[j+1] {
				flag = true
				arr[j], arr[j+1] = arr[j+1], arr[j]
			}
		}
	}
}
