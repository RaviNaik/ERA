<reponame>gooopher/go-insight
package view

import (
	"fmt"
	"testing"
)

/**
向一个已经关闭的chan读写
写操作会panic （panic: send on closed channel）
读操作,关闭前有buffer元素，可以正确读取chan的元素，并且第二个返回bool值为true，循环获取元素，直到没有buffer返回false；反之会获取chan类型的默认值，且第二个返回bool值为false；
false代表chan已关闭。因为读取操作是非阻塞的，意味着不做close判断，可以一直读取。
*/

func TestChanClosedWrite(t *testing.T) {
	ch := make(chan int)
	close(ch)

	ch <- 1
}

func TestChanClosedRead(t *testing.T) {
	ch := make(chan int, 1)
	ch <- 1
	close(ch)

	d, closed := <-ch
	fmt.Println(d, closed)
	d, closed = <-ch
	fmt.Println(d, closed)
	d, closed = <-ch
	fmt.Println(d, closed)
}
