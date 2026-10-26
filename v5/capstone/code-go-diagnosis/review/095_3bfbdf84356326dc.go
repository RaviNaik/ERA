<reponame>lxm/aliyun_assist_client<gh_stars>0
package util

import (
	"github.com/stretchr/testify/assert"
	"testing"
)
const (
	test_pri_key = `-----<KEY>`
	test_pub_key = `-----<KEY>
-----END PUBLIC KEY-----`
)

// cc03e747a6afbbcbf8be7668acfebee5

func TestRsaSignWithSha256(t *testing.T) {
//	value := RsaSign("test123", test_pri_key)
//	str_value := string(value)
//	fmt.Println(str_value)

}

func TestRsaSignWithMD5(t *testing.T) {
	value := RsaSign("changfeng", test_pri_key)
	assert.Equal(t, value, `<KEY>//<KEY>`)
}