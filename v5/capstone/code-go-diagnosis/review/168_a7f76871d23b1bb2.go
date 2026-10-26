<reponame>chanjarster/algo<gh_stars>0
package heap

import (
	"sort"
	"testing"
)

func Test_heapImpl(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
	}{
		{args: args{[]int{1, 2, 3, 4}}},
		{args: args{[]int{4, 3, 2, 1}}},
		{args: args{[]int{3, 4, 2, 1}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			h := NewHeap(false)
			for _, n := range tt.args.nums {
				h.Push(n)
			}
			if got, want := h.Len(), len(tt.args.nums); got != want {
				t.Errorf("Len() = %v, want %v", got, want)
			}
			sort.Ints(tt.args.nums)
			for _, n := range tt.args.nums {
				if got := h.Peek(); got != n {
					t.Errorf("Peek() = %v, want %v", got, n)
				}
				if got := h.Pop(); got != n {
					t.Errorf("Pop() = %v, want %v", got, n)
				}
			}
		})
	}
}
