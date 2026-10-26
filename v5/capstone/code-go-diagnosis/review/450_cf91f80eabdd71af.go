package command

import (
	"Email/pop3/response"
	"Email/socket"
)

const (
	crlf      = "\r\n"
	separator = " "
)

func Send(sck socket.Socket, cmd string, args []string) *response.Response {
	buffer := cmd
	for _, item := range args {
		buffer += separator
		buffer += item
	}
	buffer += crlf

	if sck.Write([]byte(buffer)) {
		return response.Read(sck)
	}
	return nil
}
