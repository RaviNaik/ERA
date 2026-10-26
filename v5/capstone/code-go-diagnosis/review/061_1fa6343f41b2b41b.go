<filename>03.Play-with-Algorithms/05-Binary-Search-Tree/Optional-09-Binary-Tree-Classic-Nonrecursive-Traversal/Preorder/leetcode-144_test.go
package Preorder

import (
	"fmt"
	"testing"
)

func TestLeetCode(t *testing.T) {
	root := &TreeNode{Val: 1}
	root.Right = &TreeNode{Val: 2}
	root.Right.Left = &TreeNode{Val: 3}
	fmt.Println(preorderTraversal1(root))
	fmt.Println(preorderTraversal2(root))
	fmt.Println(preorderTraversal3(root))
}
