package repository

import "github.com/shakewon/block-explorer/model/po"

type BlockRepo interface {
    /**
     * total number of blocks saved in db
     */
    Count(height,hash string) (int64, error)

    /**
     *  分页查询
     *  @index 页码
     *  @pageSize 页大小
     */
    Page(index, pageSize int,height,hash string) ([]po.Block, error)


    Save(blocks []po.Block) error

}
