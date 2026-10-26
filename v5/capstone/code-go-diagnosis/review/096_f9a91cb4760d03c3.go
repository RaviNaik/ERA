<gh_stars>0
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	_ "dcard-project/docs"
	"dcard-project/internal/controller"
	"dcard-project/internal/middleware"
	"dcard-project/internal/repository"
	"dcard-project/internal/service"
	"dcard-project/pkg/logger"
	"dcard-project/pkg/postgres"
	"dcard-project/pkg/redis"
	"github.com/gin-gonic/gin"
	_ "github.com/joho/godotenv/autoload"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

var (
	PGMaster = postgres.Config{
		Host: os.Getenv("POSTGRES_HOST"),
		User: os.Getenv("POSTGRES_USER"),
		Db:   os.Getenv("POSTGRES_DB"),
		Pwd:  <PASSWORD>("<PASSWORD>"),
		Port: os.Getenv("POSTGRES_PORT"),
	}
	PGSlave = postgres.Config{
		Host: os.Getenv("POSTGRES_SLAVE_HOST"),
		User: os.Getenv("POSTGRES_SLAVE_USER"),
		Db:   os.Getenv("POSTGRES_SLAVE_DB"),
		Pwd:  <PASSWORD>("<PASSWORD>"),
		Port: os.Getenv("POSTGRES_SLAVE_PORT"),
	}
	RdsConfig = redis.Config{
		Address: os.Getenv("REDIS_HOST") + ":" + os.Getenv("REDIS_PORT"),
		Pws:     os.Getenv("REDIS_PASSWORD"),
	}
)

var (
	r  *gin.Engine
	lg *logger.Logger
)

func main() {
	r = gin.Default()
	setupStatic(r)
	db := postgres.NewPgClient(PGMaster, PGSlave)
	rdsClient := redis.NewRedisClient(RdsConfig)
	lg = logger.NewLogger()

	middleware.Setup(rdsClient)
	repo := repository.NewUrlRepo(db, rdsClient, lg)
	urlSvc := service.NewUrlService(repo)
	controller.NewHandler(&controller.Config{Router: r, UrlSvc: urlSvc})

	GracefulRunAndShutDown()
}

func GracefulRunAndShutDown() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	srv := &http.Server{
		Addr:    ":8080",
		Handler: r,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			lg.Fatal("listen: " + err.Error())
		}
	}()

	<-ctx.Done()

	stop()
	lg.Info("shutting down gracefully, press Ctrl+C again to force")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		lg.Fatal("Server forced to shutdown: " + err.Error())
	}
	lg.Info("Server exiting")
}

func setupStatic(r *gin.Engine) {
	r.Delims("{{{", "}}}")
	r.LoadHTMLGlob("resources/views/*")
	r.Static("/asset", "./resources/asset")

	url := ginSwagger.URL(fmt.Sprintf("http://localhost:%d/swagger/doc.json", 8080))
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler, url))

	r.GET("", func(c *gin.Context) {
		c.HTML(
			http.StatusOK,
			"index.html",
			gin.H{"title": "短網址產生器"},
		)
	})
}
