<gh_stars>0
package system

import (
	"github.com/flipped-aurora/gin-vue-admin/server/model/system"
	"sync"
)

type treeSysBaseMenu struct {
	system.SysBaseMenu
	Children []treeSysBaseMenu
}

const MenuedDashboard = "dashboard"
const MenuedAbout = "about"
const MenuedAdmin = "superAdmin"
const MenuedAuthority = "authority"
const MenuedMenu = "menu"
const MenuedApi = "api"
const MenuedUser = "user"
const MenuedDictionary = "dictionary"
const MenuedDictionaryDetail = "dictionaryDetail"
const MenuedOperation = "operation"
const MenuedPerson = "person"
const MenuedExample = "example"
const MenuedExcel = "excel"
const MenuedUpload = "upload"
const MenuedBreakpoint = "breakpoint"
const MenuedCustomer = "customer"
const MenuedSimpleUploader = "simpleUploader"
const MenuedSystemTools = "systemTools"
const MenuedAutoCodeAdmin = "autoCodeAdmin"
const MenuedAutoCode = "autoCode"
const MenuedFormCreate = "formCreate"
const MenuedSystem = "system"
const MenuedHome = "https://www.gin-vue-admin.com"
const MenuedState = "state"
const MenuedAutoCodeEdit = "autoCodeEdit"

var treeSysBaseMenuData = []treeSysBaseMenu{
	{
		SysBaseMenu: system.SysBaseMenu{
			MenuLevel: 0,
			Hidden:    false,
			Name:      MenuedDashboard,
			Sort:      1,
			Component: "view/dashboard/index.vue",
			Meta:      system.Meta{Title: "仪表盘", Icon: "odometer"},
		},
	},
	{
		SysBaseMenu: system.SysBaseMenu{
			MenuLevel: 0,
			Hidden:    false,
			Name:      MenuedAbout,
			Sort:      7,
			Component: "view/about/index.vue",
			Meta:      system.Meta{Title: "关于我们", Icon: "info-filled"},
		},
	},
	{
		SysBaseMenu: system.SysBaseMenu{
			MenuLevel: 0,
			Hidden:    false,
			Path:      "admin",
			Name:      MenuedAdmin,
			Sort:      3,
			Meta:      system.Meta{Title: "超级管理员", Icon: "user"},
			Component: "view/superAdmin/index.vue",
		},
		Children: []treeSysBaseMenu{
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedAuthority,
					Sort:      1,
					Meta:      system.Meta{Title: "角色管理", Icon: "avatar"},
					Component: "view/superAdmin/authority/authority.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedMenu,
					Sort:      2,
					Meta:      system.Meta{Title: "菜单管理", Icon: "tickets", KeepAlive: true},
					Component: "view/superAdmin/menu/menu.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel:   0,
					Hidden:      false,
					Name:        MenuedApi,
					Sort:        3,
					Meta:        system.Meta{Title: "api管理", Icon: "platform", KeepAlive: true},
					Development: true,
					Component:   "view/superAdmin/api/api.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedUser,
					Sort:      4,
					Meta:      system.Meta{Title: "用户管理", Icon: "coordinate"},
					Component: "view/superAdmin/user/user.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel:   0,
					Hidden:      false,
					Name:        MenuedDictionary,
					Sort:        5,
					Meta:        system.Meta{Title: "字典管理", Icon: "notebook"},
					Development: true,
					Component:   "view/superAdmin/dictionary/sysDictionary.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel:   0,
					Hidden:      true,
					Path:        "dictionaryDetail/:id",
					Name:        MenuedDictionaryDetail,
					Sort:        1,
					Meta:        system.Meta{Title: "字典详情", Icon: "order"},
					Development: true,
					Component:   "view/superAdmin/dictionary/sysDictionaryDetail.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel:   0,
					Hidden:      false,
					Name:        MenuedOperation,
					Sort:        6,
					Meta:        system.Meta{Title: "操作历史", Icon: "pie-chart"},
					Development: true,
					Component:   "view/superAdmin/operation/sysOperationRecord.vue",
				},
			},
		},
	},
	{
		SysBaseMenu: system.SysBaseMenu{
			MenuLevel: 0,
			Hidden:    true,
			Name:      MenuedPerson,
			Sort:      4,
			Meta:      system.Meta{Title: "个人信息", Icon: "message"},
			Component: "view/person/person.vue",
		},
	},
	{
		SysBaseMenu: system.SysBaseMenu{
			MenuLevel:   0,
			Hidden:      false,
			Name:        MenuedExample,
			Sort:        6,
			Meta:        system.Meta{Title: "示例文件", Icon: "management"},
			Development: true,
			Component:   "view/example/index.vue",
		},
		Children: []treeSysBaseMenu{
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedExcel,
					Sort:      4,
					Meta:      system.Meta{Title: "excel导入导出", Icon: "takeaway-box"},
					Component: "view/example/excel/excel.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedUpload,
					Sort:      5,
					Meta:      system.Meta{Title: "媒体库（上传下载）", Icon: "upload"},
					Component: "view/example/upload/upload.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedBreakpoint,
					Sort:      6,
					Meta:      system.Meta{Title: "断点续传", Icon: "upload-filled"},
					Component: "view/example/breakpoint/breakpoint.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedCustomer,
					Sort:      7,
					Meta:      system.Meta{Title: "客户列表（资源示例）", Icon: "avatar"},
					Component: "view/example/customer/customer.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedSimpleUploader,
					Sort:      6,
					Meta:      system.Meta{Title: "断点续传（插件版）", Icon: "upload"},
					Component: "view/example/simpleUploader/simpleUploader",
				},
			},
		},
	},
	{
		SysBaseMenu: system.SysBaseMenu{
			MenuLevel: 0,
			Hidden:    false,
			Name:      MenuedSystemTools,
			Sort:      5,
			Meta:      system.Meta{Title: "系统工具", Icon: "tools"},
			Component: "view/systemTools/index.vue",
		},
		Children: []treeSysBaseMenu{
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedAutoCodeAdmin,
					Sort:      1,
					Meta:      system.Meta{Title: "自动化代码管理", Icon: "magic-stick"},
					Component: "view/systemTools/autoCodeAdmin/index.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedAutoCode,
					Sort:      1,
					Meta:      system.Meta{Title: "代码生成器", Icon: "cpu", KeepAlive: true},
					Component: "view/systemTools/autoCode/index.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedFormCreate,
					Sort:      2,
					Meta:      system.Meta{Title: "表单生成器", Icon: "magic-stick", KeepAlive: true},
					Component: "view/systemTools/formCreate/index.vue",
				},
			},
			{
				SysBaseMenu: system.SysBaseMenu{
					MenuLevel: 0,
					Hidden:    false,
					Name:      MenuedSystem,
					Sort:      3,
					Meta:      system.Meta{Title: "系统配置", Icon: "operation"},
					Component: "view/systemTools/system/system.vue",
				},
			},
		},
	},
	{
		SysBaseMenu: system.SysBaseMenu{
			MenuLevel: 0,
			Hidden:    false,
			Name:      MenuedHome,
			Sort:      0,
			Meta:      system.Meta{Title: "官方网站", Icon: "home-filled"},
		},
	},
	{
		SysBaseMenu: system.SysBaseMenu{
			MenuLevel:   0,
			Hidden:      false,
			Name:        MenuedState,
			Sort:        6,
			Meta:        system.Meta{Title: "服务器状态", Icon: "cloudy"},
			Development: true,
			Component:   "view/system/state.vue",
		},
	},
	{
		SysBaseMenu: system.SysBaseMenu{
			MenuLevel: 0,
			Hidden:    true,
			Path:      "autoCodeEdit/:id",
			Name:      MenuedAutoCodeEdit,
			Sort:      0,
			Meta:      system.Meta{Title: "自动化代码（复用）", Icon: "magic-stick"},
			Component: "view/systemTools/autoCode/index.vue",
		},
	},
}

var sysBaseMenuLock = &sync.Once{}
var sysBaseMenuData []system.SysBaseMenu
var sysBaseMenuId = uint(1)

func (t *treeSysBaseMenu) review() {
	t.ID = sysBaseMenuId
	sysBaseMenuId++
	if t.Name != "" || t.Path != "" {
		if t.Name == "" {
			t.Name = t.Path
		} else if t.Path == "" {
			t.Path = t.Name
		}
	}
	if t.Component == "" {
		t.Component = "/"
	}
}

func (t *treeSysBaseMenu) toList() {
	for _, menu := range t.Children {
		menu.ParentId = t.ID
		menu.review()
		if t.Development == true {
			menu.Development = true
		}
		if menu.Children != nil {
			menu.toList()
		} else {
			sysBaseMenuData = append(sysBaseMenuData, menu.SysBaseMenu)
		}
	}
}

func getSysBaseMenuData() []system.SysBaseMenu {
	sysBaseMenuLock.Do(func() {
		for _, menu := range treeSysBaseMenuData {
			menu.review()
			sysBaseMenuData = append(sysBaseMenuData, menu.SysBaseMenu)
			if menu.Children != nil {
				menu.toList()
			}
		}
	})
	return sysBaseMenuData
}
