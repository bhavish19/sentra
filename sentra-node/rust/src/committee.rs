use std::{collections::HashMap, fmt, sync::{Arc, RwLock}};

pub struct CommitteeMember
{
    node_id:String,
    grpc_url:String
}

impl fmt::Display for CommitteeMember {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "CommitteeMebmer: {} -- GRPC URL: {}", self.node_id,self.grpc_url)
    }
}

pub struct Committee
{
    this_node: String,
    committee: Arc<RwLock<HashMap<String, CommitteeMember>>>
}

impl Default for Committee {
    fn default()->Self
      {
        Committee
        {
            this_node:String::new(),
            committee:Arc::new(RwLock::new(HashMap::new()))
        }
      }
}

impl fmt::Display for Committee {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result 
    {
        let c: std::sync::RwLockReadGuard<'_, HashMap<String, CommitteeMember>>=self.committee.read().unwrap();
        writeln!(f, "Committee with {} members:", c.len());
        for member in c.values()
        {
            write!(f, "\t CommitteeMember: {}", member);
        }
        Ok(())
    }
}

impl Committee
{
    pub fn add(&self,node_id:&String,grpc_url:&String)
    {
        let new_member:CommitteeMember=CommitteeMember
        {
            node_id:node_id.clone(),
            grpc_url:grpc_url.clone()
        };
        self.committee.write().unwrap().insert(new_member.node_id.clone(),new_member);
    }

    pub fn establish_connections(&self)->Result<(),()>
    {
        Ok(())
    }
}