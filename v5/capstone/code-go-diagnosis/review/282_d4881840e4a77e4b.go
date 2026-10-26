package outbound

import "time"

type Attack struct {
	Host             string    `json:"host"`
	Port             int16     `json:"port"`
	AttackType       string    `json:"attack_type"`
	Packetbatchcount int64     `json:"packet_batch_count"`
	Createdtime      time.Time `json:"created_time"`
}
