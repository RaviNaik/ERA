package Teacher

import "github.com/Projects/RidingTrainingSystem/internal/pkg/entity"

// TeacherRepo interface
type TeacherRepo interface {
	SaveTeacher(teacher *entity.Teacher) error
	TeachersCount(count *uint) error
	GetTeacherByID(ID uint) (*entity.Teacher, error)
	LogTeacher(username, password string) (*entity.Teacher, error)
	TeachersOfBranchByID(BranchID uint) (*[]entity.Teacher, error)
	GetActiveLectures(BranchID, TeacherID uint) (*[]entity.Lecture, error)
	GetTodaysLecture(BranchID, TeacherID uint) (*[]entity.Lecture, error)
	ChangeImageURL(ID uint, val string) error
}
