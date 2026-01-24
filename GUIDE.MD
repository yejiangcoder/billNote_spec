# Docker 核心指令战术指南

**Project：VideoDataCatcher / billnote-backend**

> Docker 在本工程中用于承载 **BillNote / Python 转写执行环境**
> 容器名固定：`billnote-backend`
> 镜像名固定：`billnote-image`

---

## 一、容器生命周期（Lifecycle）

### 1. 创建并常驻运行（首次部署）

```powershell
docker run -d --name billnote-backend `
  -e TRANSCRIBER_TYPE=bcut `
  -v D:/Jackson_Factory/content/billnote:/data/content `
  billnote-image `
  tail -f /dev/null
```

说明：

* `-d`：后台运行
* `--name billnote-backend`：容器名固定
* `-e TRANSCRIBER_TYPE=bcut`：转写引擎环境变量
* `-v 宿主:容器`：Windows ↔ 容器挂载
* `tail -f /dev/null`：保持容器常驻待命

---

### 2. 启动 / 停止 / 重启

```powershell
docker start billnote-backend
docker stop billnote-backend
docker restart billnote-backend
```

---

### 3. 删除容器（不影响镜像）

```powershell
docker rm -f billnote-backend
```

---

## 二、状态与监控（Recon）

### 1. 查看容器状态

```powershell
docker ps
docker ps -a
```

* `ps`：仅运行中
* `ps -a`：包含已退出的容器

---

### 2. 查看实时日志

```powershell
docker logs -f billnote-backend
```

用于观察：

* Python 转写输出
* 报错 / 卡死位置

---

### 3. 资源占用监控

```powershell
docker stats
```

---

## 三、执行与切入（Execution）

### 1. 直接让容器执行一次脚本（Java 后台本质）

```powershell
docker exec billnote-backend python main.py <URL>
```

---

### 2. 进入容器内部（调试）

```powershell
docker exec -it billnote-backend /bin/bash
```

常用检查路径：

```bash
ls /data/content
```

---

## 四、镜像管理（Logistics）

### 1. 查看本地镜像

```powershell
docker images
```

---

### 2. 构建镜像（backend 目录下）

```powershell
docker build -t billnote-image .
```

---

### 3. 删除镜像

```powershell
docker rmi <IMAGE_ID>
```

---

## 五、常用战术组合（Cheat Sheet）

### 1. 完整重置容器（配置 / 挂载异常）

```powershell
docker rm -f billnote-backend

docker run -d --name billnote-backend `
  -e TRANSCRIBER_TYPE=bcut `
  -v D:/Jackson_Factory/content/billnote:/data/content `
  billnote-image `
  tail -f /dev/null
```

---

### 2. 校验挂载是否成功

```powershell
docker exec billnote-backend ls /data/content
```

---

### 3. 清理 Docker 空间（慎用）

```powershell
docker system prune -a
```

---

## 六、当前实例状态

```text
容器 ID:
d2ede942edcf5a9fc89f805cb32178c2b12f9d46bc55dfe50c22ee1fed935127
```

