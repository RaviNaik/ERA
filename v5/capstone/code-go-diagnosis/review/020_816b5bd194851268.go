<filename>src/golang-master/models/usermodel.go
package models

import (
	"github.com/jmoiron/sqlx"
	"golang-master/lang"
	"golang-master/generallib"
)
type User struct {
    Id	  	int64 `json:"id"`
    Email   string `json:"email"`
	Token 	string `json:"token"`
	Role    string `json:"role"`
}

type ReqLogin struct {
    Email   string `json:"email" validate:"required"`
	Password string `json:"password" validate:"required"`
}

// Admin Login
func Login(db *sqlx.DB, reqlogin *ReqLogin)  (*User, string) {
	email := reqlogin.Email
	password := reqlogin.Password
	var user User

	err := db.Get(&user,"Select id,email,'' as token,role from users where email = ? and password = ?",email,generallib.GetMD5Hash(password))
	if err != nil {
		return &user, lang.Get("inavlid_username_or_password")
	}

	validToken, err := generallib.GenerateJWT(user.Email, user.Role)
	if err != nil {
		return &user, lang.Get("failed_to_generate_token")
	}
	user.Token = validToken;
	return &user, ""	
}