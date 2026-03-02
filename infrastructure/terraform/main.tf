# TradeMaster Infrastructure as Code
# Manages Railway project, services, and dependencies

terraform {
  required_providers {
    railway = {
      source  = "terraform-community-providers/railway"
      version = "~> 0.4"
    }
  }
}

variable "railway_token" {
  type      = string
  sensitive = true
}

variable "environment" {
  type    = string
  default = "production"
}

provider "railway" {
  token = var.railway_token
}

# Main project
resource "railway_project" "trademaster" {
  name = "trademaster-${var.environment}"
}

# Backend service
resource "railway_service" "backend" {
  project_id = railway_project.trademaster.id
  name       = "backend"
}

# Frontend service
resource "railway_service" "frontend" {
  project_id = railway_project.trademaster.id
  name       = "frontend"
}

# PostgreSQL database
resource "railway_service" "postgres" {
  project_id = railway_project.trademaster.id
  name       = "postgres"
}

# Redis cache
resource "railway_service" "redis" {
  project_id = railway_project.trademaster.id
  name       = "redis"
}

output "project_id" {
  value = railway_project.trademaster.id
}
