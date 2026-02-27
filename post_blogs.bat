@echo off
REM Daily Blog Auto-Poster — TenantStack + PhysicianPad
REM Runs Monday-Friday at 8am PST via Windows Task Scheduler

cd /d C:\Users\dansc\projects\llm-router

echo [%date% %time%] Starting daily blog post... >> post_blogs.log
python auto_blog_post.py >> post_blogs.log 2>&1
echo [%date% %time%] Done. >> post_blogs.log
