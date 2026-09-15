#!/bin/sh
set -eu
mkdir -p /etc/nginx
cat > /etc/nginx/upstreams.conf <<'EOF'
upstream backend {
    least_conn;
    server service:8080;
}
EOF
refresh() {
    ips=$(getent hosts service | awk '{print $1}' | sort -u)
    {
        echo "upstream backend {"
        echo "    least_conn;"
        if [ -z "$ips" ]; then
            echo "    server service:8080;"
        else
            for ip in $ips; do
                echo "    server ${ip}:8080;"
            done
        fi
        echo "}"
    } > /etc/nginx/upstreams.conf
}

refresh
nginx -g "daemon off;" &
NGINX_PID=$!
while kill -0 "$NGINX_PID" 2>/dev/null; do
    refresh
    nginx -s reload 2>/dev/null || true
    sleep 2
done
wait "$NGINX_PID"
