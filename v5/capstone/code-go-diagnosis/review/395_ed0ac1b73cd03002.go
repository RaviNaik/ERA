package main

import (
	"testing"
)

func TestBalanceLeftLeftRotatation(t *testing.T) {
	tree := &Tree{}
	tree.Insert(30)
	tree.Insert(20)
	tree.Insert(10)

	if tree.Root.Value == 20 {
		t.Log("Value of Root element PASSED, is 20")
	} else {
		t.Errorf("Value of Root element FAILED, should be 20 but is %d", tree.Root.Value)
	}

	if tree.Root.Left.Value == 10 {
		t.Log("Value of Root -> Left element PASSED, is 10")
	} else {
		t.Errorf("Value of Root -> Left element FAILED, should be 10 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Right.Value == 30 {
		t.Log("Value of Root -> Right element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right element FAILED, should be 30 but is %d", tree.Root.Right.Value)
	}
}

func TestBalanceLeftRightRotatation(t *testing.T) {
	tree := &Tree{}
	tree.Insert(30)
	tree.Insert(10)
	tree.Insert(20)

	if tree.Root.Value == 20 {
		t.Log("Value of Root element PASSED, is 20")
	} else {
		t.Errorf("Value of Root element FAILED, should be 20 but is %d", tree.Root.Value)
	}

	if tree.Root.Left.Value == 10 {
		t.Log("Value of Root -> Left element PASSED, is 10")
	} else {
		t.Errorf("Value of Root -> Left element FAILED, should be 10 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Right.Value == 30 {
		t.Log("Value of Root -> Right element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right element FAILED, should be 30 but is %d", tree.Root.Right.Value)
	}
}

func TestBalanceRightRightRotatation(t *testing.T) {
	tree := &Tree{}
	tree.Insert(10)
	tree.Insert(20)
	tree.Insert(30)

	if tree.Root.Value == 20 {
		t.Log("Value of Root element PASSED, is 20")
	} else {
		t.Errorf("Value of Root element FAILED, should be 20 but is %d", tree.Root.Value)
	}

	if tree.Root.Left.Value == 10 {
		t.Log("Value of Root -> Left element PASSED, is 10")
	} else {
		t.Errorf("Value of Root -> Left element FAILED, should be 10 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Right.Value == 30 {
		t.Log("Value of Root -> Right element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right element FAILED, should be 30 but is %d", tree.Root.Right.Value)
	}
}

func TestBalanceRightLeftRotatation(t *testing.T) {
	tree := &Tree{}
	tree.Insert(10)
	tree.Insert(30)
	tree.Insert(20)

	if tree.Root.Value == 20 {
		t.Log("Value of Root element PASSED, is 20")
	} else {
		t.Errorf("Value of Root element FAILED, should be 20 but is %d", tree.Root.Value)
	}

	if tree.Root.Left.Value == 10 {
		t.Log("Value of Root -> Left  element PASSED, is 10")
	} else {
		t.Errorf("Value of Root -> Left  element FAILED, should be 10 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Right.Value == 30 {
		t.Log("Value of Root -> Right  element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right  element FAILED, should be 30 but is %d", tree.Root.Right.Value)
	}
}

func TestBalanceBigRotatation(t *testing.T) {
	tree := &Tree{}
	tree.Insert(40)
	tree.Insert(20)
	tree.Insert(10)
	tree.Insert(25)
	tree.Insert(30)

	t.Log("After adding in sequence 40, 20. 10, 25 and 30:")
	if tree.Root.Value == 20 {
		t.Log("Value of Root element PASSED, is 20")
	} else {
		t.Errorf("Value of Root element FAILED, should be 20 but is %d", tree.Root.Value)
	}

	if tree.Root.Left.Value == 10 {
		t.Log("Value of Root -> Left element PASSED, is 10")
	} else {
		t.Errorf("Value of Root -> Left element FAILED, should be 10 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Right.Value == 30 {
		t.Log("Value of Root -> Right element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right element FAILED, should be 30 but is %d", tree.Root.Right.Value)
	}

	if tree.Root.Right.Left.Value == 25 {
		t.Log("Value of Root -> Right -> Left element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right -> Left element FAILED, should be 30 but is %d", tree.Root.Right.Value)
	}

	if tree.Root.Right.Right.Value == 40 {
		t.Log("Value of Root -> Right -> Right element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right -> Right element FAILED, should be 30 but is %d", tree.Root.Right.Value)
	}

	t.Log("After adding 22:")
	tree.Insert(22)

	if tree.Root.Value == 25 {
		t.Log("Value of Root element PASSED, is 25")
	} else {
		t.Errorf("Value of Root element FAILED, should be 25 but is %d", tree.Root.Value)
	}

	if tree.Root.Left.Value == 20 {
		t.Log("Value of Root -> Left element PASSED, is 20")
	} else {
		t.Errorf("Value of Root -> Left element FAILED, should be 20 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Left.Left.Value == 10 {
		t.Log("Value of Root -> Left -> Left element PASSED, is 10")
	} else {
		t.Errorf("Value of Root -> Left -> Left element FAILED, should be 10 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Left.Right.Value == 22 {
		t.Log("Value of Root -> Left -> Right element PASSED, is 22")
	} else {
		t.Errorf("Value of Root -> Left -> Right element FAILED, should be 22 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Right.Value == 30 {
		t.Log("Value of Root -> Right element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right element FAILED, should be 30 but is %d", tree.Root.Right.Value)
	}

	if tree.Root.Right.Right.Value == 40 {
		t.Log("Value of Root -> Right -> Right element PASSED, is 40")
	} else {
		t.Errorf("Value of Root -> Right -> Right element FAILED, should be 40 but is %d", tree.Root.Right.Value)
	}

	t.Log("After adding 50:")
	tree.Insert(50)

	if tree.Root.Value == 25 {
		t.Log("Value of Root element PASSED, is 20")
	} else {
		t.Errorf("Value of Root element FAILED, should be 25 but is %d", tree.Root.Value)
	}

	if tree.Root.Left.Value == 20 {
		t.Log("Value of Root -> Left element PASSED, is 10")
	} else {
		t.Errorf("Value of Root -> Left element FAILED, should be 20 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Left.Left.Value == 10 {
		t.Log("Value of Root -> Left -> Left element PASSED, is 10")
	} else {
		t.Errorf("Value of Root -> Left -> Left element FAILED, should be 10 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Left.Right.Value == 22 {
		t.Log("Value of Root -> Left -> Right element PASSED, is 10")
	} else {
		t.Errorf("Value of Root -> Left -> Right element FAILED, should be 22 but is %d", tree.Root.Left.Value)
	}

	if tree.Root.Right.Value == 40 {
		t.Log("Value of Root -> Right element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right element FAILED, should be 40 but is %d", tree.Root.Right.Value)
	}

	if tree.Root.Right.Left.Value == 30 {
		t.Log("Value of Root -> Right -> Left element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right -> Left element FAILED, should be 30 but is %d", tree.Root.Right.Value)
	}

	if tree.Root.Right.Right.Value == 50 {
		t.Log("Value of Root -> Right -> Right element PASSED, is 30")
	} else {
		t.Errorf("Value of Root -> Right -> Right element FAILED, should be 50 but is %d", tree.Root.Right.Value)
	}
}

func TestFind(t *testing.T) {
	tree := &Tree{}
	tree.Insert(40)
	tree.Insert(20)
	tree.Insert(10)
	tree.Insert(25)
	tree.Insert(30)
	tree.Insert(22)
	tree.Insert(50)

	node, ok := tree.Find(30)
	if ok && node.Value == 30 {
		t.Log("Find in tree PASSED")
	} else {
		t.Error("Find in tree FAILED")
	}

	node, ok = tree.Find(76)
	if !ok {
		t.Log("Not find in tree PASSED")
	} else {
		t.Error("Not find in tree FAILED")
	}
}
