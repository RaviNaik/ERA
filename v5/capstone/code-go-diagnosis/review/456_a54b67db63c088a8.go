<filename>constants/chat.go
package constants

import (
	"github.com/gorilla/websocket"
	"gopkg.in/fatih/set.v0"
	"sync"
)

const (
	JWT_TOKEN_KEY      = "token"
	CONNECT_TYPE_USER  = 1
	CONNECT_TYPE_GROUP = 2
)

/**
消息体
*/
type MessageConstant struct {
	Id       int64  `json:"id,omitempty" form:"id"`             //消息ID
	Userid   int64  `json:"userid,omitempty" form:"userid"`     //谁发的
	Cmd      int    `json:"cmd,omitempty" form:"cmd"`           //群聊还是私聊
	TargetID int64  `json:"targetid,omitempty" form:"targetid"` //对端用户ID/群ID
	Media    int    `json:"media,omitempty" form:"media"`       //消息按照什么样式展示
	Content  string `json:"content,omitempty" form:"content"`   //消息的内容
	Pic      string `json:"pic,omitempty" form:"pic"`           //预览图片
	Url      string `json:"url,omitempty" form:"url"`           //服务的URL
	Memo     string `json:"memo,omitempty" form:"memo"`         //简单描述
	Amount   int    `json:"amount,omitempty" form:"amount"`     //其他和数字相关的
}

//本核心在于形成userid和Node的映射关系
type NodeConstant struct {
	Conn      *websocket.Conn
	DataQueue chan []byte   //并行转串行的队列,Conn 是一个IO型的资源 存在竞争关系
	GroupSets set.Interface //第三方包,可以快速获取 并集 交集 差集等,用以存储群组信息
}

//映射关系表
type ClientMapConstant struct {
	sync.Map
}

func (r *ClientMapConstant) add(userID int64, node *NodeConstant) {
	r.Store(userID, node)
}

func (r *ClientMapConstant) delete(userID int64) (*NodeConstant, bool) {
	value, ok := r.LoadAndDelete(userID)
	if ok {
		return value.(*NodeConstant), true
	}
	return nil, false
}
