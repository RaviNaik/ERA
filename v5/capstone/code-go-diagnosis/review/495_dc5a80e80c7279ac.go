<filename>vggame/public/config/item.go<gh_stars>0
package iconfig

type ItemConf struct {
	Id                 int32
	Name               string
	FirstType          int32
	SecondType         int32
	Quality            int32
	Lv                 int32
	UseType            int32
	HeroRandom         int64
	HeroForce          int64
	HeroBrains         int64
	HeroPolitics       int64
	HeroCharm          int64
	SkillExp           int64
	BookExp            int64
	BeautyIntimacy     int64
	BeautyCharm        int64
	FamilyMoney        int64
	FamilyContribution int64
	Attribute          int32
	Reward             int32
	HeroId             int32
}

type IItemConfig interface {
	IConfig
	GetConf(id int32) *ItemConf
}
