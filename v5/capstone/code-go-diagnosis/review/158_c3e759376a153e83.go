<reponame>yusabana-sandbox/golang-practice
package chapter6

import "fmt"

type Stringer interface {
	String() string
}
type MyFunc func() string
type MyString string
type MyNumber int

// 関数にインターフェースを実装
func (f MyFunc) String() string { return f() }

// stringにインターフェースを実装
func (s MyString) String() string { return "fooです" }

// number(int)にインターフェースを実装
func (n MyNumber) String() string { return fmt.Sprintf("%dです", n) }

// Stringerを実装
type Hex int

func (h Hex) String() string {
	return fmt.Sprintf("%x", int(h))
}

// Hex2もStringerを実装していることになる
type Hex2 struct{ Hex }

func DoSamples() {
	// fmt.Stringer というインターフェースを実装
	var s fmt.Stringer = MyFunc(func() string { return "hi" })

	fmt.Println(s)
	fmt.Println(s.String())

	// インターフェース変数.(型) でインターフェース型の値を任意の型にキャスト
	var v interface{}
	v = 100
	num, ok := v.(int)
	fmt.Println(num, ok)
	str, ok := v.(string)
	fmt.Println(str, ok)

	// インターフェースを実装した型を使ってみる
	var myF = MyFunc(func() string { return "あいうえお" })
	var myS = MyString("ほげ")
	var myN = MyNumber(333)
	fmt.Println(myF.String(), myS.String(), myN.String())

	showType(myF)
	showType(myS)
	showType(myN)

	// hexのサンプル
	var ss Stringer
	hh := Hex(100)
	ss = hh
	fmt.Println(ss) // fmt.Printlnを使っているので ss.String() ではなくても自動で String() が呼ばれる

	hh2 := Hex2{hh}
	ss = hh2
	fmt.Println(ss)
}

func showType(s interface{}) {
	fmt.Println(s)
	switch v := s.(type) {
	case MyFunc:
		println("FUNC", v)
		fmt.Println("FUNC", v)
	case MyString:
		println("STRING", v)     // ほげ
		fmt.Println("STRING", v) // fooです  ここが異なるのはString()メソッドが定義されていてfmtパッケージはそれを利用するため
	case MyNumber:
		println("NUMBER", v)
		fmt.Println("NUMBER", v)
	}
}
