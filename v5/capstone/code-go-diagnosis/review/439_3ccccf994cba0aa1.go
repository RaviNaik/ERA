<gh_stars>0
package utils

import "fmt"

type FamilyAccount struct {
	// 菜单项
	key string
	// 控制是否退出for
	loop bool
	// 账户余额
	balance float64
	// 每次收支金额
	money float64
	// 收支说明
	note string
	// 收支详情，当有收支时进行拼接处理
	details string
	// 初始details长度
	flag int
}

// 编写工厂模式构造方法，返回一个*FamilyAccount实例
func InitFamilyAccount() *FamilyAccount {
	return &FamilyAccount{
		key:     "",
		loop:    true,
		balance: 0.00,
		money:   0.00,
		// 收支说明
		note: "",
		// 收支详情，当有收支时进行拼接处理
		details: "收支\t账户金额\t收支金额\t说      明\n",
		// 初始details长度
		flag: len("收支\t账户金额\t收支金额\t说      明\n"),
	}
}

// 显示明细
func (account *FamilyAccount) showDetails() {
	fmt.Println("**********当前收支明细记录**********")
	if account.flag == len(account.details) {
		fmt.Println("当前没有收支明细")
	} else {
		fmt.Println(account.details)
	}
}

// 登记收入
func (account *FamilyAccount) income() {
	fmt.Println("本次收入金额：")
	fmt.Scanln(&account.money)
	account.balance += account.money
	fmt.Println("本次收入说明：")
	fmt.Scanln(&account.note)
	account.details += fmt.Sprintf("收入\t%v\t%v\t%v\n", account.balance, account.money, account.note)
	fmt.Println("当前账户余额：", account.balance)
}

// 显示支出
func (account *FamilyAccount) pay() {
	fmt.Println("本次支出金额：")
	fmt.Scanln(&account.money)
	if account.money > account.balance {
		fmt.Println("余额不足")
	} else {
		account.balance -= account.money
		fmt.Println("本次支出说明：")
		fmt.Scanln(&account.note)
		account.details += fmt.Sprintf("支出\t%v\t%v\t%v\n", account.balance, account.money, account.note)
		fmt.Println("当前账户余额：", account.balance)
	}
}

// 退出
func (account *FamilyAccount) exit() {
	account.loop = false
}

// 显示主菜单
func (account *FamilyAccount) MainShow() {
	for {
		fmt.Println("**********家庭收支管理**********")
		fmt.Println("          1 收支明细")
		fmt.Println("          2 登记收入")
		fmt.Println("          3 登记支出")
		fmt.Println("          4 退出")
		fmt.Println("请选择1-4菜单")

		fmt.Scanln(&account.key)

		switch account.key {
		case "1":
			account.showDetails()
		case "2":
			account.income()
		case "3":
			account.pay()
		case "4":
			account.exit()
		default:
			fmt.Println("请输入正确的选项")
		}
		if !account.loop {
			break
		}
	}
	fmt.Println("已退出软件")
}
